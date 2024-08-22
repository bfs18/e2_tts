### Implementation of [E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS](https://arxiv.org/abs/2406.18009)

Inference results and some discussion can be found [here](https://github.com/lucidrains/e2-tts-pytorch/issues/25)

Usage:

build filelist and phoneset with scripts/build_ljspeech.py and build_filelist.py

update the filelist path and phoneset path config/e2_tts.yaml

train model: python3 train.py -c config/e2_tts.yaml

inference with rfwave vocoder: download [acoustic ckpt trained with LJSpeech](https://drive.google.com/file/d/1NjaxSuCQSDBy2gys9cRcbpvQcRW-szNs/view?usp=sharing) and the [vocoder ckpt trained with LibriTTS](https://drive.google.com/file/d/1IQNXAAVRTtr9P8Gc-CoPeRIJ_l_O4y38) 
and inference with inference.py. Synthesized samples from this checkpoint can be found [here](https://drive.google.com/file/d/1EpNC_7LqE9-52U7sn1ohchWaQn54sAvr/view?usp=sharing)

```
python3 inference.py --test_txt tests/test.txt \
    --aco_model_dir /path/to/e2_tts-cfg_2_cond2-bf16-large_batch \
    --voc_model_dir /path/to/rfwave-libritts-24k \
    --phoneset /path/to/phoneset.th \
    --save_dir syn_e2-08-22 --sr 24000
```
