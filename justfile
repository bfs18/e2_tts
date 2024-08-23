python:
    #!/bin/bash
    # Check if .venv directory exists; if not, create a virtual environment
    if [ ! -d ".venv" ]; then \
        python -m venv .venv; \
    fi
    # Activate the virtual environment
    source .venv/bin/activate
    pip install -r requirements.txt

setup:
    # Step 1: Download and extract the LJSpeech dataset
    wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
    tar -xvjf LJSpeech-1.1.tar.bz2
    # Step 3: Download and extract rfwave ckpt
    # wget https://drive.google.com/file/d/1IQNXAAVRTtr9P8Gc-CoPeRIJ_l_O4y38 -O rfwave-libritts-24k.tar.gz
    tar -xvzf rfwave-libritts-24k.tar.gz
    # Step 3: Download and extract e2_tts ckpt
    # wget https://drive.google.com/file/d/1NjaxSuCQSDBy2gys9cRcbpvQcRW-szNs/view?usp=sharing -O e2_tts-cfg_2_cond2-bf16-large_batch.tar.gz
    tar -xvzf e2_tts-cfg_2_cond2-bf16-large_batch.tar.gz

data:
    #!/bin/bash
    source .venv/bin/activate
    # Step 1: Run build_ljspeech script
    python scripts/build_ljspeech.py && \
    # Step 2: Run build_filelist script with specified parameters
    python scripts/build_filelist.py --wav_dir LJSpeech-1.1/wavs --transcription_dir LJSpeech-1.1/transcription --filelist LJSpeech-1.1/filelist

test:
    #!/bin/bash
    source .venv/bin/activate
    python inference.py --aco_model_dir ./e2_tts-cfg_2_cond2-bf16-large_batch --voc_model_dir ./rfwave-libritts-24k --phoneset tests/phoneset.th --test_txt tests/test.txt  --save_dir ./test_output --ref_audio tests/LJ025-0077.wav --sr 22050

play: python setup data test
