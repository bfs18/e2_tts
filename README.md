### Implementation of [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.18009)

Inference results can be found [here](https://github.com/lucidrains/e2-tts-pytorch/issues/25)

Usage:

build filelist and phoneset with scripts/build_ljspeech.py and build_filelist.py

update the filelist path and phoneset path config/e2_tts.yaml

train model: python3 train.py -c config/e2_tts.yaml

inference with rfwave vocoder: download the [vocoder ckpt trained with LibriTTS](https://drive.google.com/file/d/1IQNXAAVRTtr9P8Gc-CoPeRIJ_l_O4y38) and inference with inference.py
