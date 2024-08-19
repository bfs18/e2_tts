import math
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
import torchaudio

from torch import nn
from rfwave.feature_extractors import FeatureExtractor
from rfwave.heads import FourierHead
from rfwave.dit import Backbone
from rfwave.multi_band_processor import PQMFProcessor, STFTProcessor
from rfwave.input import InputAdaptor


class RectifiedFlow(nn.Module):
    def __init__(self, backbon: Backbone, head: FourierHead,
                 num_steps=10, feature_loss=False, wave=False, num_bands=8, p_uncond=0., guidance_scale=1.):
        super().__init__()
        self.backbone = backbon
        self.head = head
        self.N = num_steps
        self.feature_loss = feature_loss
        # wave: normal -(stft)-> freq_noise -> NN -> freq_feat -(istft)-> wave -> loss
        # freq: normal -> NN -> freq_feat -> loss
        self.wave = wave
        self.equalizer = wave
        self.stft_norm = not wave
        self.stft_loss = False
        self.phase_loss = False
        self.overlap_loss = True
        self.num_bands = num_bands
        self.num_bins = self.head.n_fft // 2 // self.num_bands
        self.left_overlap = 8
        self.right_overlap = 8
        self.overlap = self.left_overlap + self.right_overlap
        self.cond_mask_right_overlap = True
        self.prev_cond = False
        self.parallel_uncond = True
        self.time_balance_loss = True
        self.noise_alpha = 0.1
        self.cfg = guidance_scale > 1.
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        assert self.backbone.output_channels == self.head.n_fft // self.num_bands + 2 * self.overlap
        assert self.prev_cond == self.backbone.prev_cond
        assert self.wave ^ self.stft_norm
        assert self.right_overlap >= 1  # at least one to deal with the last dimension of fft feature.
        if self.stft_norm:
            self.stft_processor = STFTProcessor(self.head.n_fft + 2)
        if self.equalizer:
            self.eq_processor = PQMFProcessor(subbands=8, taps=124, cutoff_ratio=0.071)

    def get_subband(self, S, i):
        if i.numel() > 1:
            i = i[0]
        S = torch.stack(torch.chunk(S, 2, dim=1), dim=-1)
        if i == -1:
            sS = S.new_zeros((S.shape[0], (self.num_bins + self.overlap) * 2, S.shape[2]))
        else:
            pS = F.pad(S, (0, 0, 0, 0, self.left_overlap, self.right_overlap - 1), mode='constant')
            sS = pS[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap]
            sS = torch.cat([sS[..., 0], sS[..., 1]], dim=1)
        return sS

    def place_subband(self, sS, i):
        if i.numel() > 1:
            i = i[0]
        S = sS.new_zeros([sS.size(0), self.head.n_fft // 2 + self.overlap, sS.size(2), 2])
        rsS, isS = torch.chunk(sS, 2, dim=1)
        S[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap, :, 0] = rsS
        S[:, i * self.num_bins: (i + 1) * self.num_bins + self.overlap, :, 1] = isS
        S = S[:, self.left_overlap: S.size(1) - self.right_overlap + 1]
        return torch.cat([S[..., 0], S[..., 1]], dim=1)

    def get_joint_subband(self, S):
        S = torch.stack(torch.chunk(S, 2, dim=1), dim=-1)
        S = F.pad(S, (0, 0, 0, 0, self.left_overlap, self.right_overlap - 1), mode='circular')
        # for version before 2.1
        # S = torch.cat([S[:, S.size(1) - self.left_overlap:], S, S[:, :self.right_overlap - 1]], dim=1)
        assert S.size(1) == self.num_bins * self.num_bands + self.overlap
        S = S.unfold(1, self.num_bins + self.overlap, self.num_bins)  # shape (batch_size, num_bands, seq_len, 2, band_dim)
        S = S.permute(0, 1, 4, 2, 3)  # shape (batch_size, num_bands, band_dim, seq_len, 2)
        S = torch.cat([S[..., 0], S[..., 1]], dim=2)
        S = S.reshape(S.size(0), -1, S.size(-1))
        return S

    def place_joint_subband(self, S):
        def _get_subband(s, i):
            if i == self.num_bands - 1:
                return s[:, self.left_overlap: s.size(1) - self.right_overlap + 1]
            else:
                return s[:, self.left_overlap: s.size(1) - self.right_overlap]
        assert S.size(1) == self.num_bands * (self.num_bins + self.overlap) * 2
        sS_ri = torch.chunk(S, self.num_bands * 2, dim=1)
        sS_r = [_get_subband(s, i) for i, s in enumerate(sS_ri[0::2])]
        sS_i = [_get_subband(s, i) for i, s in enumerate(sS_ri[1::2])]
        return torch.cat([torch.cat(sS_r, dim=1), torch.cat(sS_i, dim=1)], dim=1)

    def mask_cond(self, cond):
        cond = torch.stack(torch.chunk(cond, 2, dim=1), dim=-1)
        cond[:, cond.size(1) - self.right_overlap:] = 0.
        return torch.cat([cond[..., 0], cond[..., 1]], dim=1)

    def get_z0(self, mel, bandwidth_id):
        if bandwidth_id.numel() > 1:
            bandwidth_id = bandwidth_id[0]
        if self.wave:
            nf = mel.shape[2] if self.head.padding == "same" else (mel.shape[2] - 1)
            r = torch.randn([mel.shape[0], self.head.hop_length * nf], device=mel.device)
            rf = self.stft(r)
            z0 = self.get_subband(rf, bandwidth_id)
        else:
            r = torch.randn([mel.shape[0], self.head.n_fft + 2, mel.shape[2]], device=mel.device)
            z0 = self.get_subband(r, bandwidth_id)
        return z0

    def get_joint_z0(self, mel):
        if self.wave:
            nf = mel.shape[2] if self.head.padding == "same" else (mel.shape[2] - 1)
            r = torch.randn([mel.shape[0], self.head.hop_length * nf], device=mel.device)
            rf = self.stft(r)
            z0 = self.get_joint_subband(rf)
        else:
            r = torch.randn([mel.shape[0], self.head.n_fft + 2, mel.shape[2]], device=mel.device)
            z0 = self.get_joint_subband(r)
        z0 = z0.reshape(z0.size(0) * self.num_bands, z0.size(1) // self.num_bands, z0.size(2))
        return z0

    def get_eq_norm_stft(self, audio):
        if self.equalizer:
            audio = self.eq_processor.project_sample(audio.unsqueeze(1)).squeeze(1)
        S = self.stft(audio)
        if self.stft_norm:
            S = self.stft_processor.project_sample(S)
        return S

    def get_z1(self, audio, bandwidth_id):
        if bandwidth_id.numel() > 1:
            bandwidth_id = bandwidth_id[0]
        S = self.get_eq_norm_stft(audio)
        z1 = self.get_subband(S, bandwidth_id)
        if self.prev_cond:
            cond_band = self.get_subband(S, bandwidth_id - 1)
            if self.cond_mask_right_overlap:
                cond_band = self.mask_cond(cond_band)
        else:
            cond_band = None
        return z1, cond_band

    def get_joint_z1(self, audio):
        S = self.get_eq_norm_stft(audio)
        z1 = self.get_joint_subband(S)
        z1 = z1.reshape(z1.size(0) * self.num_bands, z1.size(1) // self.num_bands, z1.size(2))
        return z1

    def get_wave(self, x):
        if self.stft_norm:
            x = self.stft_processor.return_sample(x)
        x = self.istft(x)
        if self.equalizer:
            x = self.eq_processor.return_sample(x.unsqueeze(1)).squeeze(1)
        return x

    def get_train_tuple(self, mel, audio_input):
        if self.prev_cond or not self.parallel_uncond:
            t = torch.rand((mel.size(0),), device=mel.device)
            bandwidth_id = torch.tile(torch.randint(0, self.num_bands, (), device=mel.device), (mel.size(0),))
            bandwidth_id = torch.ones([mel.shape[0]], dtype=torch.long, device=mel.device) * bandwidth_id
            z0 = self.get_z0(mel, bandwidth_id)
            z1, cond_band = self.get_z1(audio_input, bandwidth_id)
            mel = torch.cat([mel, cond_band], 1) if self.prev_cond else mel
        else:
            t = torch.rand((mel.size(0),), device=mel.device).repeat_interleave(self.num_bands, 0)
            z0 = self.get_joint_z0(mel)
            z1 = self.get_joint_z1(audio_input)
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
            mel = torch.repeat_interleave(mel, self.num_bands, 0)
        t_ = t.view(-1, 1, 1)
        z_t = t_ * z1 + (1. - t_) * z0
        target = z1 - z0
        return mel, bandwidth_id, (z_t, t, target)

    def get_pred(self, z_t, t, mel, bandwidth_id, encodec_bandwidth_id=None):
        pred = self.backbone(z_t, t, mel, bandwidth_id, encodec_bandwidth_id)
        return pred

    @torch.no_grad()
    def sample_ode_subband(self, mel, band, bandwidth_id,
                           encodec_bandwidth_id=None, N=None, keep_traj=False):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.N
        traj = []  # to store the trajectory
        dt = 1. / N
        if self.prev_cond or not self.parallel_uncond:
            assert band is not None
            assert bandwidth_id is not None
            # get z0 must be called before pre-processing mel
            _, z0 = self.get_z0(mel, torch.tensor(0, dtype=torch.long, device=mel.device))
            mel = torch.cat([mel, band], 1) if self.prev_cond else mel
        else:
            assert band is None
            assert bandwidth_id is None
            bandwidth_id = torch.tile(torch.arange(self.num_bands, device=mel.device), (mel.size(0),))
            z0 = self.get_joint_z0(mel)  # get z0 must be called before pre-processing mel
            mel = torch.repeat_interleave(mel, self.num_bands, 0)

        z = z0.detach()
        fs = (z.size(0) // self.num_bands, z.size(1) * self.num_bands, z.size(2))
        ss = z.shape
        for i in range(N):
            t = torch.ones(z.size(0)) * i / N
            if self.cfg:
                mel_ = torch.cat([mel, torch.ones_like(mel) * mel.mean(dim=(0, 2), keepdim=True)], dim=0)
                (z_, t_, bandwidth_id_) = [torch.cat([v] * 2, dim=0) for v in (z, t, bandwidth_id)]
                pred = self.get_pred(z_, t_.to(mel.device), mel_, bandwidth_id_, encodec_bandwidth_id)
                pred, uncond_pred = torch.chunk(pred, 2, dim=0)
                pred = uncond_pred + self.guidance_scale * (pred - uncond_pred)
            else:
                pred = self.get_pred(z, t.to(mel.device), mel, bandwidth_id, encodec_bandwidth_id)
            if self.wave:
                if self.prev_cond or not self.parallel_uncond:
                    pred = self.place_subband(pred, bandwidth_id)
                    pred = self.stft(self.istft(pred))
                    pred = self.get_subband(pred, bandwidth_id)
                    z = z.detach() + pred * dt
                else:
                    pred = self.place_joint_subband(pred.reshape(fs))
                    pred = self.stft(self.istft(pred))
                    pred  = self.get_joint_subband(pred).reshape(ss)
                    z = z.detach() + pred * dt
            else:
                z = z.detach() + pred * dt
            if i == N - 1 or keep_traj:
                traj.append(z.detach())
        return traj

    def combine_subbands(self, traj):
        assert len(traj) == self.num_bands

        def _reshape(x):
            return torch.stack(torch.chunk(x, 2, dim=1), dim=-1)

        def _combine(l):
            c_x = []
            for i, x_i in enumerate(l):
                if i == len(l) - 1:
                    x_i = _reshape(x_i)[:, self.left_overlap: x_i.size(1) // 2 - self.right_overlap + 1]
                else:
                    x_i = _reshape(x_i)[:, self.left_overlap: x_i.size(1) // 2 - self.right_overlap]
                c_x.append(x_i)
            c_x = torch.cat(c_x, dim=1)
            return torch.cat([c_x[..., 0], c_x[..., 1]], dim=1)

        c_traj = []
        for traj_bands in zip(*traj):
            c_traj.append(_combine(traj_bands))

        return c_traj

    def sample_ode(self, mel, encodec_bandwidth_id=None, N=None, keep_traj=False):
        traj = []
        if self.prev_cond or not self.parallel_uncond:
            band = mel.new_zeros((mel.shape[0], 2 * (self.num_bins + self.overlap), mel.shape[2]), device=mel.device)
            for i in range(self.num_bands):
                bandwidth_id = torch.ones([mel.shape[0]], dtype=torch.long, device=mel.device) * i
                traj_i = self.sample_ode_subband(
                    mel, band, bandwidth_id, encodec_bandwidth_id=encodec_bandwidth_id, N=N, keep_traj=keep_traj)
                band = traj_i[-1]
                if self.prev_cond:
                    band = torch.zeros_like(band)
                elif self.cond_mask_right_overlap:
                    band = self.mask_cond(band)
                traj.append(traj_i)
            traj = self.combine_subbands(traj)
        else:
            traj_f = self.sample_ode_subband(
                mel, None, None,
                encodec_bandwidth_id=encodec_bandwidth_id, N=N, keep_traj=keep_traj)
            traj = [self.place_joint_subband(tt.reshape(tt.size(0) // self.num_bands, -1, tt.size(2)))
                    for tt in traj_f]
        return [self.get_wave(tt) for tt in traj]

    def stft(self, wave):
        S = self.head.get_spec(wave.float()) / np.sqrt(self.head.n_fft).astype(np.float32)
        return torch.cat([S.real, S.imag], dim=1).type_as(wave)

    def istft(self, S):
        S = S * np.sqrt(self.head.n_fft).astype(np.float32)
        r, i = torch.chunk(S.float(), 2, dim=1)
        c = r + 1j * i
        return self.head.get_wave(c).type_as(S)


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        head: FourierHead,
        input_adaptor: InputAdaptor = None,
        task: str = "voc",
        sample_rate: int = 24000,
        initial_learning_rate: float = 2e-4,
        feature_loss: bool = False,
        wave: bool = False,
        num_bands: int = 8,
        guidance_scale: float = 1.,
        p_uncond: float = 0.2,
        num_warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "head", "input_adaptor"])

        self.task = task
        self.feature_extractor = feature_extractor
        self.input_adaptor = input_adaptor
        self.reflow = RectifiedFlow(
            backbone, head, feature_loss=feature_loss, wave=wave, num_bands=num_bands,
            guidance_scale=guidance_scale, p_uncond=p_uncond)
        assert num_bands == backbone.num_bands
