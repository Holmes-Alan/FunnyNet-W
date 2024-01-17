# 1. Install `BYOL-A` and download the pretrained model "AudioNTT2020-BYOLA-64x96d2048.pth".
mkdir ext
cd ext
git clone git@github.com:nttcslab/byol-a.git
cd byol-a
pip install -r requirements.txt
cd ../..
mv ext/byol-a ext/byol_a
mv ext/byol-a/pretrained_weights/AudioNTT2020-BYOLA-64x96d2048.pth ./models
