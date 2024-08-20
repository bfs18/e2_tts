import warnings
import torch
import re

import torchaudio
import yaml
import soundfile
import rfwave
import string

from pathlib import Path
from argparse import ArgumentParser
from g2p_en import G2p
from collections import OrderedDict
from rfwave.helpers import plot_spectrogram_to_numpy
from matplotlib import pyplot

g2p = G2p()

torch.set_float32_matmul_precision('high')


def remove_space_around_punctuation(phones):
    new_phones = []
    for i in range(len(phones)):
        if phones[i] in string.punctuation:
            if len(new_phones) and new_phones[-1] == ' ':
                del new_phones[-1]
        elif i > 0 and phones[i - 1] in string.punctuation and phones[i] == ' ':
            continue
        new_phones.append(phones[i])
    return new_phones


def add_bos_eos(phones):
    new_phones = []
    if phones[-1] not in string.punctuation:
        new_phones = phones + ['.']
    else:
        new_phones = phones[:]
    return ["<BOS>"] + new_phones


def load_config(config_yaml):
    with open(config_yaml, 'r') as stream:
        config = yaml.safe_load(stream)
    return config


def create_instance(config):
    for k, v in config['init_args'].items():
        if isinstance(v, dict) and 'class_path' in v and 'init_args' in v:
            config['init_args'][k] = create_instance(v)
    return eval(config['class_path'])(**config['init_args'])


def load_model(model_dir, device, last=False):
    config_yaml = Path(model_dir) / 'config.yaml'
    if last:
        ckpt_fp = list(Path(model_dir).rglob("last.ckpt"))
        if len(ckpt_fp) == 0:
            raise ValueError(f"No checkpoint found in {model_dir}")
        elif len(ckpt_fp) > 1:
            warnings.warn(f"More than 1 checkpoints found in {model_dir}")
            ckpt_fp = sorted([fp for fp in ckpt_fp], key=lambda x: x.stat().st_ctime)[-1:]
        ckpt_fp = ckpt_fp[0]
        print(f'using last ckpt form {str(ckpt_fp)}')
    else:
        ckpt_fp = [fp for fp in list(Path(model_dir).rglob("*.ckpt")) if 'last' not in fp.stem]
        ckpt_fp = sorted(ckpt_fp, key=lambda x: int(re.search('_step=(\d+)_', x.stem).group(1)))[-1]
        print(f'using best ckpt form {str(ckpt_fp)}')

    config = load_config(config_yaml)
    exp = create_instance(config['model'])

    model_dict = torch.load(ckpt_fp, map_location='cpu')
    for k in list(model_dict['state_dict'].keys()):
        if k.startswith('rvm.') or k.startswith('melspec_loss.'):
            del model_dict['state_dict'][k]

    exp.load_state_dict(model_dict['state_dict'])
    exp.eval()
    exp.to(device)
    return exp


def get_phones(text_line, phoneset):
    phones = g2p(text_line)
    phones = [p for p in phones if p in phoneset]
    phones = remove_space_around_punctuation(phones)
    phones = add_bos_eos(phones)
    return phones


def parse_text(test_txt, phoneset):
    lines = [l.strip() for l in Path(test_txt).open()]
    text_dict = dict()
    for l in lines:
        fields = l.split('|')
        k = fields[0]
        text = fields[-1]
        text_dict[k] = get_phones(text, phoneset)
    return text_dict


def save_fig(spec, save_fp):
    pyplot.figure(figsize=(12, 3))
    pyplot.imshow(spec, aspect="auto")
    pyplot.tight_layout()
    pyplot.savefig(save_fp)
    pyplot.close()


def tts(aco_model_dir, voc_model_dir, text_lines, ref_audio, ref_text, phone2id, save_dir, sr, N=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aco_exp = load_model(aco_model_dir, device=device, last=True)
    aco_config_yaml = Path(aco_model_dir) / 'config.yaml'
    aco_config = load_config(aco_config_yaml)
    voc_exp = load_model(voc_model_dir, device=device, last=True)
    voc_config_yaml = Path(voc_model_dir) / 'config.yaml'
    voc_config = load_config(voc_config_yaml)
    assert ref_audio.ndim == 2 and ref_audio.size(0) == 1
    ref_mel = aco_exp.feature_extractor(ref_audio.to(device))
    # ref_mel = aco_exp.mel_processor.project_sample(ref_mel) ref mel is not projected before feeding into InputAdaptor
    # ref_token_ids = torch.tensor([phone2id[str(tk)] for tk in ref_text])
    pi_kwargs = {'ctx_start': torch.tensor([0], dtype=torch.long).to(device),
                 'ctx_length': torch.tensor([ref_mel.size(2)], dtype=torch.long).to(device)}
    for k, line in list(sorted(text_lines.items()))[:10]:
        full_line = ref_text + line[1:]  # only 1 <BOS>
        token_ids = torch.tensor([phone2id[str(tk)] for tk in full_line])
        token_ids = token_ids.unsqueeze(0).to(device)
        phone_info = [token_ids, ref_mel]
        pi_kwargs['num_tokens'] = torch.tensor([token_ids.size(1)], dtype=torch.long).to(device)
        text = aco_exp.input_adaptor(*phone_info)
        dur_kwargs = aco_exp.infer_dur(text, **pi_kwargs)
        pi_kwargs.update(**dur_kwargs)
        print('synthesizing', k, 'num_tokens', token_ids.size(1), 'num_frames', pi_kwargs['out_length'].item())
        mel_hat = aco_exp.sample_ode(text, N=N, **pi_kwargs)[-1]
        mel_hat = aco_exp.mel_processor.return_sample(mel_hat)
        spec = plot_spectrogram_to_numpy(mel_hat.detach().cpu().numpy()[0])
        fig_fp = Path(save_dir) / f'{k}-full.png'
        save_fig(spec, fig_fp)
        mel_hat = torch.exp(mel_hat).log10()  # voc training used log 10
        audio_hat = voc_exp.reflow.sample_ode(mel_hat, N=10)[-1]
        audio_hat = audio_hat.detach().cpu().numpy()
        soundfile.write(Path(save_dir) / f'{k}-full.wav', audio_hat.T, samplerate=sr, subtype='PCM_16')
        mel_hat_syn = mel_hat[..., ref_mel.size(2):]
        audio_hat_syn = voc_exp.reflow.sample_ode(mel_hat_syn, N=10)[-1]
        audio_hat_syn = audio_hat_syn.detach().cpu().numpy()
        soundfile.write(Path(save_dir) / f'{k}-syn.wav', audio_hat_syn.T, samplerate=sr, subtype='PCM_16')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--aco_model_dir', type=str, required=True)
    parser.add_argument('--voc_model_dir', type=str, required=True)
    parser.add_argument('--phoneset', type=str, required=True)
    parser.add_argument('--test_txt', type=str, required=True)
    parser.add_argument('--ref_audio', type=str, default="tests/LJ025-0077.wav")
    parser.add_argument('--ref_text', type=str, default="Their food is provided for them,")
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, choices=[22050, 24000], default=24000)

    args = parser.parse_args()
    Path(args.save_dir).mkdir(exist_ok=True)

    phoneset = torch.load(args.phoneset)
    phoneset = ["_PAD_"] + phoneset
    phone2id = dict([(p, i) for i, p in enumerate(phoneset)])

    Path(args.save_dir).mkdir(exist_ok=True)
    text_lines = parse_text(args.test_txt, phoneset)
    ref_text = get_phones(args.ref_text, phoneset)
    assert Path(args.ref_audio).exists()
    ref_audio, sr = torchaudio.load(args.ref_audio)
    assert sr == args.sr
    tts(args.aco_model_dir, args.voc_model_dir, text_lines, ref_audio, ref_text, phone2id, args.save_dir, sr=args.sr)
