import numpy as np
import torch
import torchaudio
import kaldiio
import random

from dataclasses import dataclass
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
from rfwave.bucket import DynamicBucketingDataset, DynamicBucketingSampler
from torch.nn import functional as F

torch.set_num_threads(1)


def get_num_tokens(tokens, padding_value=0):
    # take 1 pad in to consideration.
    num_tokens = (tokens != padding_value).sum(1)
    num_tokens += (tokens[:, -1] == padding_value)
    return num_tokens


def get_exp_length(num_tokens, token_exp_scale):
    max_val = num_tokens.max()
    num_tokens = torch.where(num_tokens == max_val, num_tokens, num_tokens - 1)
    length = torch.round(num_tokens * token_exp_scale).long()
    return length


def get_exp_scale(num_tokens, length):
    max_val = num_tokens.max()
    token_exp_scale = torch.where(
        num_tokens == max_val, length.float() / num_tokens.float(), length.float() / (num_tokens.float() - 1))
    return token_exp_scale


@dataclass
class DataConfig:
    filelist_path: str
    batch_size: int
    num_workers: int
    sampling_rate: int = 24000
    num_samples: int = 65280
    cache: bool = False
    task: str = "voc"
    hop_length: int = None
    padding: str = None
    phoneset: str = None
    segment: bool = True
    min_context: int = 50
    max_context: int = 300
    max_duration: float = 100
    max_cuts: int = 32
    num_buckets: int = 20
    drop_last: bool = False
    quadratic_duration: Optional[float] = None
    filter_max_duration: Optional[float] = None
    random_batch_every_epoch: bool = False


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        if cfg.task == "e2e":
            dataset = E2ETTSCtxDataset(cfg, train=train)
            collate_fn = e2e_tts_ctx_collate
        else:
            raise ValueError(f"Unknown task: {cfg.task}")
        if cfg.segment:
            dataloader = DataLoader(
                dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train,
                pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        else:
            batch_sampler = DynamicBucketingSampler(dataset, random_batch_every_epoch=cfg.random_batch_every_epoch)
            dataloader = DataLoader(
                dataset, batch_sampler=batch_sampler, num_workers=cfg.num_workers,
                pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class E2ETTSCtxDataset(DynamicBucketingDataset):
    def __init__(self, cfg: DataConfig, train: bool):
        super(E2ETTSCtxDataset, self).__init__(
            filelist_path=cfg.filelist_path,
            max_duration=cfg.max_duration,
            max_cuts=cfg.max_cuts,
            num_buckets=cfg.num_buckets,
            shuffle=train,
            drop_last=cfg.drop_last,
            quadratic_duration=cfg.quadratic_duration,
            filter_max_duration=cfg.filter_max_duration)
        assert cfg.task == "e2e"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None
        self.min_context = cfg.min_context
        self.max_context = cfg.max_context
        self.gain = -3.

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        k, audio_fp, phone_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(phone_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{self.gain:.2f}"]])
            if self._cache is not None:
                self._cache[k] = (y, token_ids)
        else:
            y, token_ids = self._cache[k]

        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        num_frames = y.size(1) // self.hop_length + (1 if self.padding == "center" else 0)
        token_ids = token_ids.detach().clone()
        exp_scale = num_frames / token_ids.size(0)

        if y.size(-1) > self.min_context * self.hop_length * 2:
            max_context = np.minimum(num_frames // 2, self.max_context)
            ctx_n_frame = np.random.randint(self.min_context, max_context)
            ctx_start_frame = np.random.randint(0, num_frames - ctx_n_frame - 1)
        else:
            ctx_n_frame = num_frames // 2 - 1
            ctx_start_frame = 0 if np.random.rand() < 0.5 else ctx_n_frame

        # get context
        ctx_start = ctx_start_frame * self.hop_length
        ctx_end = (ctx_start_frame + ctx_n_frame) * self.hop_length
        y_ctx = y[:, ctx_start: ctx_end]
        ctx_n_frame = ctx_n_frame + 1 if self.padding == 'center' else ctx_n_frame

        assert ctx_start_frame + ctx_n_frame <= num_frames
        return y[0], (token_ids, len(token_ids), y_ctx[0], ctx_start_frame, ctx_n_frame, exp_scale)


def e2e_tts_ctx_collate(data):
    y_lens = [d[0].size(0) for d in data]
    max_y_len = max(y_lens)
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    max_num = max(num_phones)
    y = torch.zeros([len(data), max_y_len], dtype=torch.float)
    y_ctx = [ti[2] for ti in token_info]
    max_ctx_len = max([y.size(0) for y in y_ctx])
    token_ids = torch.zeros([len(data), max_num], dtype=torch.long)
    y_ctx_pad = torch.zeros([len(data), max_ctx_len], dtype=torch.float)
    for i, (ti, _, ctx, *_) in enumerate(token_info):
        y[i, :y_lens[i]] = data[i][0]
        token_ids[i, :ti.size(0)] = ti
        y_ctx_pad[i, :ctx.size(0)] = ctx
    num_tokens_ = torch.tensor([ti[1] for ti in token_info])
    num_tokens = get_num_tokens(token_ids)
    assert num_tokens_.sum() + len(data) == num_tokens.sum() + (token_ids[:, -1] != 0).sum()
    ctx_start = torch.tensor([ti[3] for ti in token_info])
    ctx_n_frame = torch.tensor([ti[4] for ti in token_info])
    exp_scale_ = torch.tensor([ti[5] for ti in token_info])
    #TODO: a special case, not correct. #tok + 1 = max #tok
    special_exp_scale = torch.round(num_tokens_ * exp_scale_) / (num_tokens_ + 1)
    exp_scale = torch.where(num_tokens_ == num_tokens_.max() - 1, special_exp_scale, exp_scale_)
    # assert torch.all(torch.tensor([yl // 256 + 1 for yl in y_lens]) == get_exp_length(num_tokens, exp_scale)), \
    #     (f'num_tokens_: {num_tokens_}, num_tokens:{num_tokens}, frames: {[yl // 256 + 1 for yl in y_lens]}, '
    #      f'exp_scal1:{torch.round(exp_scale_ * num_tokens_).long()} '
    #      f'exp_scale2: {get_exp_length(num_tokens, exp_scale)}')
    return y, [token_ids, num_tokens, y_ctx_pad, ctx_start, ctx_n_frame, exp_scale]
