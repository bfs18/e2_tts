import math
import wandb

import numpy as np
import pytorch_lightning as pl
import torch
import transformers
import torch.nn.functional as F

from rfwave.feature_extractors import FeatureExtractor
from rfwave.helpers import plot_spectrogram_to_numpy, save_figure_to_numpy, plot_attention_to_numpy
from rfwave.dit import Backbone
from rfwave.attention import score_mask, sequence_mask
from rfwave.input import InputAdaptor
from rfwave.multi_band_processor import DurationProcessor, MeanVarProcessor
from rfwave.dataset import get_exp_length
from rfwave.e2e_duration import E2EDuration, DurModel


def sequence_mask_with_ctx(length, ctx_start=None, ctx_length=None, max_length=None):
    non_padding = sequence_mask(length + 1, max_length)
    non_padding = non_padding[:, :-1]  # 1 padding frame got trained for layer norm
    if ctx_length is None or ctx_start is None:
        return non_padding
    else:
        assert torch.all(ctx_start + ctx_length < length)
        if max_length is None:
            max_length = length.max()
        non_ctx = torch.arange(0, max_length, device=length.device).unsqueeze(0)
        non_ctx = torch.logical_or(
            non_ctx < ctx_start.unsqueeze(1), non_ctx >= (ctx_start + ctx_length).unsqueeze(1))
        return torch.logical_and(non_ctx, non_padding)


class VocosExp(pl.LightningModule):
    # noinspection PyUnusedLocal
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone: Backbone,
        input_adaptor: InputAdaptor,
        sample_rate: int,
        initial_learning_rate: float,
        num_warmup_steps: int
    ):
        """
        Args:
            feature_extractor (FeatureExtractor): An instance of FeatureExtractor to extract features from audio signals.
            backbone (Backbone): An instance of Backbone model.
            head (FourierHead):  An instance of Fourier head to generate spectral coefficients and reconstruct a waveform.
            sample_rate (int): Sampling rate of the audio signals.
            initial_learning_rate (float): Initial learning rate for the optimizer.
            num_warmup_steps (int): Number of steps for the warmup phase of learning rate scheduler. Default is 0.
            mel_loss_coeff (float, optional): Coefficient for Mel-spectrogram loss in the loss function. Default is 45.
            mrd_loss_coeff (float, optional): Coefficient for Multi Resolution Discriminator loss. Default is 1.0.
            pretrain_mel_steps (int, optional): Number of steps to pre-train the model without the GAN objective. Default is 0.
            decay_mel_coeff (bool, optional): If True, the Mel-spectrogram loss coefficient is decayed during training. Default is False.
            evaluate_utmos (bool, optional): If True, UTMOS scores are computed for each validation run.
            evaluate_pesq (bool, optional): If True, PESQ scores are computed for each validation run.
            evaluate_periodicty (bool, optional): If True, periodicity scores are computed for each validation run.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["feature_extractor", "backbone", "standalone_alignment", "input_adaptor"])

        self.feature_extractor = feature_extractor
        self.backbone = torch.compile(backbone)
        self.input_adaptor = input_adaptor

        self.validation_step_outputs = []
        self.automatic_optimization = True
        self.dur_processor = MeanVarProcessor(1)
        self.mel_processor = MeanVarProcessor(feature_extractor.dim)
        self.dim = feature_extractor.dim
        self.N = 20

        self.standalone_dur = E2EDuration(DurModel(self.input_adaptor.dim, 2), output_exp_scale=True)
        self.dur_processor = DurationProcessor()
        self.standalone_dur_start_step = 10000
        self.train_dur = True

    def configure_optimizers(self):
        gen_params = [
            {"params": self.backbone.parameters()},
            {"params": self.input_adaptor.parameters()},
            {"params": self.standalone_dur.parameters()}
        ]

        opt_gen = torch.optim.AdamW(gen_params, lr=self.hparams.initial_learning_rate, betas=(0.8, 0.9))

        max_steps = self.trainer.max_steps  # Max steps per optimizer
        scheduler_gen = transformers.get_cosine_schedule_with_warmup(
            opt_gen, num_warmup_steps=self.hparams.num_warmup_steps, num_training_steps=max_steps,
        )

        return (
            [opt_gen], [{"scheduler": scheduler_gen, "interval": "step"}],)

    def skip_nan(self, optimizer):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = torch.isfinite(param.grad).all()
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("detected inf or nan values in gradients. not updating model parameters")
            optimizer.zero_grad()

    def on_before_optimizer_step(self, optimizer):
        # Note: `unscale` happens after the closure is executed, but before the `on_before_optimizer_step` hook.
        self.skip_nan(optimizer)
        self.clip_gradients(optimizer, gradient_clip_val=100., gradient_clip_algorithm="norm")

    def forward(self, z_t, t, features, **kwargs):
        pred = self.backbone(z_t, t, features, **kwargs)
        return pred

    def get_z0(self, text, length):
        r = torch.randn([text.shape[0], self.dim, length], device=text.device)
        return r

    def get_z1(self, mel):
        return mel

    def get_train_tuple(self, text, mel):
        t = torch.rand([text.shape[0]], device=text.device)
        t_ = t.view(-1, 1, 1)
        z0 = self.get_z0(text, mel.size(2))
        z1 = self.get_z1(mel)
        z_t = t_ * z1 + (1. - t_) * z0
        target = z1 - z0
        return t, z_t, target

    def process_context(self, phone_info):
        pi_kwargs = {}
        ctx_kwargs = {}
        assert len(phone_info) == 6
        phone_info[2] = self.feature_extractor(phone_info[2])
        # num_tokens * epx_scale to get num_frames
        # length = torch.round(phone_info[1] * phone_info[5]).long()
        length = get_exp_length(phone_info[1], phone_info[5])
        pi_kwargs['num_tokens'] = phone_info[1]
        pi_kwargs['ctx_start'] = phone_info[3]
        pi_kwargs['ctx_length'] = phone_info[4]
        pi_kwargs['token_exp_scale'] = phone_info[5]
        ctx_kwargs['length'] = length
        ctx_kwargs['ctx_start'] = phone_info[3]
        ctx_kwargs['ctx_length'] = phone_info[4]
        phone_info = [phone_info[0], phone_info[2]]
        return phone_info, pi_kwargs, ctx_kwargs

    def compute_dur_loss(self, text, standalone_attn, **kwargs):
        num_tokens = kwargs['num_tokens']
        ref_length = kwargs['ctx_length']
        dur_out = self.standalone_dur(text, num_tokens, ref_length)
        token_exp_scale = kwargs['token_exp_scale']
        token_exp_scale = self.dur_processor.project_sample(token_exp_scale)
        loss = F.mse_loss(dur_out, token_exp_scale)
        if self.global_step < self.standalone_dur_start_step:
            return loss * 0.  # all weights are used.
        else:
            return loss

    def training_step(self, batch, *args, **kwargs):
        audio_input, phone_info = batch
        phone_info, pi_kwargs, ctx_kwargs = self.process_context(phone_info)
        text = self.input_adaptor(*phone_info)
        mel = self.feature_extractor(audio_input, **kwargs)
        mel = self.mel_processor.project_sample(mel)
        ctx_mask = sequence_mask_with_ctx(**ctx_kwargs)
        t, z_t, target = self.get_train_tuple(text, mel)
        kwargs.update(pi_kwargs)
        pred = self.forward(z_t, t, text, **kwargs)
        pred = torch.where(ctx_mask.unsqueeze(1), pred, target)
        rf_loss = F.mse_loss(pred, target)
        dur_loss = self.compute_dur_loss(text, **pi_kwargs)
        loss = rf_loss + dur_loss
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/rf_loss", rf_loss)
        self.log("train/dur_loss", dur_loss)
        return rf_loss

    def sample_ode(self, text, N=None, keep_traj=False, **kwargs):
        if N is None:
            N = self.N
        z0 = self.get_z0(text, kwargs['out_length'])
        z = z0
        ts = torch.linspace(0, 1, N + 1, device=text.device)
        traj = []
        for i, t in enumerate(ts[:-1]):
            dt = ts[i + 1] - t
            t_ = torch.ones(text.size(0), device=text.device) * t
            pred = self.forward(z, t_, text, **kwargs)
            z = z.detach() + pred * dt
            if i == N - 1 or keep_traj:
                traj.append(z.detach())
        return traj

    def infer_dur(self, text, **kwargs):
        num_tokens = kwargs['num_tokens']
        ref_length = kwargs['ctx_length']
        dur_out = self.standalone_dur(text, num_tokens, ref_length)
        dur_out = self.dur_processor.return_sample(dur_out)
        length = get_exp_length(num_tokens, dur_out)
        return {'out_length': length.clamp(0).max(), 'token_exp_scale': dur_out.clamp(0)}

    def validation_step(self, batch, *args, **kwargs):
        audio_input, phone_info = batch
        phone_info, pi_kwargs, ctx_kwargs = self.process_context(phone_info)
        with torch.no_grad():
            text = self.input_adaptor(*phone_info)
            mel = self.feature_extractor(audio_input, **kwargs)
            mel = self.mel_processor.project_sample(mel)
            kwargs.update(**pi_kwargs)
            kwargs['out_length'] = mel.size(2)
            # out_length, token_exp_scale will be replaced.
            aod_kwargs = (self.infer_dur(text, **pi_kwargs)
                          if self.global_step > self.standalone_dur_start_step * 1.5 else {})
            kwargs.update(**aod_kwargs)
            mel_hat = self.sample_ode(text, **kwargs)[-1]

        mel_hat_interp = F.interpolate(mel_hat, size=mel.shape[2:])
        mel_loss = F.l1_loss(mel_hat_interp, mel)

        output = {
            "mel_loss": mel_loss,
            "audio_input": audio_input[0].float(),
            "mel_hat": mel_hat[0],
        }
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        mel_loss = torch.stack([x["mel_loss"] for x in outputs]).mean()

        if self.global_rank == 0:
            audio_in, mel_hat = outputs[0]['audio_input'], outputs[0]['mel_hat']
            mel_hat = self.mel_processor.return_sample(mel_hat.unsqueeze(0)).squeeze(0)
            mel_target = self.feature_extractor(audio_in.unsqueeze(0)).squeeze(0)
            metrics = {"val_loss": mel_loss, "val/mel_loss": mel_loss}
            self.logger.log_metrics(metrics, step=self.global_step)
            self.logger.experiment.log(
                {"valid/mel_in": wandb.Image(plot_spectrogram_to_numpy(mel_target.data.cpu().numpy())),
                 "valid/mel_hat": wandb.Image(plot_spectrogram_to_numpy(mel_hat.data.cpu().numpy()))},
                step=self.global_step)
        self.log("val_loss", mel_loss, sync_dist=True, logger=False)
        self.validation_step_outputs.clear()
