# FunnyNet-W (Multimodal Learning of Funny Moments in Videos in the Wild)

[![Open TxST in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1RO_gZBqSHoNWt-lo8oaw5NBI2E3dXASi?usp=sharing)<br>

By Zhi-Song Liu, Robin Courant and Vicky Kalogeiton

This repo only provides simple testing codes, pre-trained models, and the network strategy demo.

We present FunnyNet-W, a versatile and efficient framework for funny moment detection in the video.

### [Project Page](https://www.lix.polytechnique.fr/vista/projects/2024_ijcv_liu/) | [Paper](https://arxiv.org/pdf/2401.04210.pdf) | [Data](https://drive.google.com/drive/folders/1ZM6agmEnheiyP0IIrD3Fc7DOubjyu5eO?usp=share_link)

# BibTex

        @InProceedings{funnynet-w,
            author = {Liu, Zhi-Song and Courant, Robin and Kalogeiton, Vicky},
            title = {FunnyNet-W: Multimodal Learning of Funny Moments in Videos in the Wild},
            booktitle = {International Journal of Computer Vision},
            year = {2024},
            pages={},
            doi={}
        }
  
## Dependencies

Python 3.8
OpenCV library
Pytorch 1.12.0
CUDA 11.3

## Environment setup

1. Clone code to your local computer.
```sh
git clone https://github.com/robincourant/FunnyNet.git
cd FunnyNet
```

2. Create working environment.
```sh
conda create --name funnynet -y python=3.8
conda activate funnynet
```

1. Install the dependencies.
```sh
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

1. Run the setup script to intsall all the dependencies.
```
./setup.sh
```

1. Modify in `ext/TimeSformer/timesformer/models/vit_utils.py`
```
from torch._six import container_abcs --> import collections.abc as container_abcs
```

1. Comment `ext/TimeSformer/timesformer/models/resnet_helper.py`
```
from torch.nn.modules.linear import _LinearWithBias
```

1. Download friends data:
```
gdown https://drive.google.com/drive/folders/1ZM6agmEnheiyP0IIrD3Fc7DOubjyu5eO -O ./data --folder
```
Note: label files are strutured as follow: [season, episode, funny-label, start, end]

The dataset directory is organized as followed:
```
FunnyNet-data/
└── tv_show_name/
    ├── audio/
    │   ├── diff/              # `.wav` files with stereo channel difference
    │   ├── embedding/         # `.pt` files with audio embedding vectors
    │   ├── laughter/          # `.pickle` files with laughter timecodes
    │   ├── laughter_segment/  # `.wav` files with detected laughters
    │   ├── left/              # `.wav` files with the surround left channel
    │   └── raw/               # `.wav` files with extracted raw audio from videos
    ├── laughter/              # `.pk` files with laughter labels
    ├── sub/                   # `.pk` files with subtitles
    ├── episode/               # `.mkv` files with videos
    ├── audio_split/           # `.wav` files with audio 8 seconds windows
    │   ├── test_8s/
    │   ├── train_8s/
    │   └── validation_8s/
    ├── video_split/           # `.mp4` files with video 8 seconds windows
    │   ├── test_8s/
    │   ├── train_8s/
    │   └── validation_8s/
    └── sub_split/             # `.pk` files with subtitles 8 seconds windows
        ├── sub_test_8s.pk
        ├── sub_train_8s.pk
        └── sub_validation_8s.pk
```
Note: we cannot provide audio and video data for obvious copyright issues.

## FunnyNet

### Data processing

Split audio, subtitles and videos into segments of n seconds (default 8 seconds):
```sh
python data_processing/mask_audio.py DATA_DIR/audio/raw DATA_DIR/audio/laughter DATA_DIR/audio/processed
python data_processing/audio_processing.py DATA_DIR/audio/raw DATA_DIR/laughter/xx.pk DATA_DIR/audio_split
python data_processing/sub_processing.py DATA_DIR/sub DATA_DIR/laughter/xx.pk DATA_DIR/sub_split
python data_processing/video_processing.py DATA_DIR/episode DATA_DIR/laughter/xx.pk DATA_DIR/video_split
```

### Training

1. Train multimodality with audio and vision
```sh
python funnynet/train.py model.batch_size=BATCH_SIZE xp_name=XP_NAME data.data_dir=DATA_DIR model=avf-timesformer-byol-lstm data=avf-timesformer-byol-lstm
```

### Testing

1. Test multimodality with audio and vision
```sh
python funnynet/evaluate.py
```


## Laughter detection

There is 4 scripts:

- `laughter_detection/scripts/extract_audio.py`: extracts from video files contained in `episode/` corresponding audio tracks and saves them in `audio/raw/` .

- `laughter_detection/scripts/detect_laughter.py`: detects laughters from audio files in `audio/raw/` and saves laughter timecodes as `.pickle` files in `audio/laughter/`.

- `laughter_detection/scripts/extract_laughter.py`: extracts from raw audio segments in `audio/raw/` each detected laughter in `audio/laughter/` and saves them in `audio/laughter_segment/`.

- `laughter_detection/scripts/evaluate_laughters.py`: given directories of predicted and ground-truth laughter files (`.pickle`), compare them and compute metrics.

- 
