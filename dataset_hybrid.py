from PIL import Image
import os.path as osp
import random
import re
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torchvision
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from transformers import AutoImageProcessor
import os


from ext.byol_a.byol_a.augmentations import PrecomputedNorm

warnings.filterwarnings("ignore")
os.environ['CURL_CA_BUNDLE'] = ''

def augment(img, rot, flip_H, flip_V):
    if flip_V < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if flip_H < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    return img



class Audio_vision_dataset_v2_all(Dataset):  
    def __init__(self, data_dir, llama2_dir, time_step, cfg):
        self.frame_rate = 24.0

        # ############# audio parameters #################
        self.resample = 16000
        self.n_mels = 64
        self.fixed_length = 500
        self.stats = [-5.4919195, 5.0389895]
        self.melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        self.normalizer = PrecomputedNorm(self.stats)

        # ############# vision parameters ################
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_size = 224
        self.num_frame = 16
        self.time_step = time_step

        self.transform_data = transforms.Compose(
            [
                transforms.RandomCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        # ############# sub parameters ################
        self.tokenizer = AutoTokenizer.from_pretrained(llama2_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # ############# label file #######################
        self.classes_for_all_imgs = []
        self.annotation = []
        self.sub = []
        self.video = []
        self.audio = []
        
        self.annotation_1 = pd.read_pickle(osp.join(data_dir, 'laughter/train_final.pk'))
        self.sub_1 = pd.read_pickle(osp.join(data_dir, 'Train/sub/train_whisper_8s/sub_train_8s.pk'))
        self.video_1 = osp.join(data_dir, 'Train/video/train_final_8s')
        self.audio_1 = osp.join(data_dir, 'Train/audio_new/train_final_8s')

        for i in range(len(self.annotation_1)):
            a = [1, i, self.annotation_1[i][2]]
            self.annotation.append(a)
            sub = self.sub_1[i][-1]
            sub = re.sub("\n", " ", sub)
            sub = re.sub("-", " ", sub)
            b = [1, i, sub]
            self.sub.append(b)
            c = [1, i, self.video_1]
            self.video.append(c)
            d = [1, i, self.audio_1]
            self.audio.append(d)
            clas_id = self.annotation_1[i][2] 
            self.classes_for_all_imgs.append(clas_id)
            

        self.melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.resample, n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.annotation)

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs

    def __getitem__(self, index):
        # ########## sample ground truth label #####################
        label = self.annotation[index][2]  

        # ############## sample audio data #########################
        melspectogram_db = self.load_audio(index)

        # ############## sample video data #########################
        sequence = self.load_image(index, label)

        # ############## sample sub data #########################
        sub = self.sub[index][2]
        if sub == "":
            sub += " "
        # finish augmentation
        sub_tokens = self.tokenizer.encode_plus(
            text=sub,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=64,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = sub_tokens["input_ids"].squeeze(0)
        attention_mask = sub_tokens["attention_mask"].squeeze(0)

        return sequence, melspectogram_db, input_ids, attention_mask, label

    def load_audio(self, index):
        audio_path = self.audio[index][2]
        no = self.annotation[index][1]

        wav_name = osp.join(audio_path, str(no).zfill(5) + ".wav")
        soundData, sample_rate = torchaudio.load(wav_name)
        if soundData.shape[0] > 1:
            soundData = torch.mean(soundData, dim=0).unsqueeze(0)

        resampler = T.Resample(sample_rate, self.resample)
        soundData = resampler(soundData)
        frame_length = soundData.shape[1] / self.resample

        if frame_length > self.time_step:
            start_time = torch.randint(int(frame_length - self.time_step), (1,)) * self.resample
            end_time = start_time + self.time_step * self.resample
            soundData = soundData[:, start_time:end_time]
        else:
            tmp = torch.zeros(1, self.resample * self.time_step)
            tmp[:, : soundData.shape[1]] = soundData
            soundData = tmp
        
        if torch.rand(1) > 0.5:
            soundData = soundData + torch.randn((1, soundData.shape[1])) * 0.1
            shift_idx = torch.randint(16, (1,)) + 1
            soundData = torch.roll(soundData, shifts=self.resample//shift_idx.item(), dims=1)

        melspectogram = self.normalizer(
            (
                self.melspectrogram_transform(soundData) + torch.finfo(torch.float).eps
            ).log()
        )

        return melspectogram  

    def load_image(self, index, label):
        video_path = self.video[index][2]
        no = self.annotation[index][1]

        path = osp.join(video_path, str(no).zfill(5) + ".mp4")

        video = torchvision.io.read_video(path)
        frame_rate = video[2]["video_fps"]
        len_time = int(float(video[0].shape[0]) / frame_rate)
        
        start_time = 0
        end_time = 0
        #=================== step frame_rate=========================
        if len_time > self.time_step:
            start_time = int(torch.randint(int(len_time - self.time_step), (1,)) * frame_rate)
            end_time = start_time + int(self.time_step * frame_rate)
        else:
            start_time = 0
            end_time = int(len_time * frame_rate)

        video = video[0][start_time:end_time, :, :, :]

        # --------------------------------------------------------------------------------------------------------#
        sequence = torch.zeros((3, self.num_frame, self.img_size, self.img_size))

        step = int(float(video.shape[0]) / float(self.num_frame))
        if step >= 1:
            table = torch.arange(0, video.shape[0], step)
            for i in range(self.num_frame):
                idx = table[i]
                img = video[idx : idx + 1, :, :, :].squeeze(0).numpy().astype("uint8")
                img = Image.fromarray(img)
                img = self.transform_data(img).unsqueeze(1)
                sequence[:, i : i + 1, :, :] = img
                if torch.rand(1) > 0.5:
                    sequence = sequence + torch.randn(sequence.shape) * 0.1
                    shift_idx = torch.randint(7, (1,)) + 1
                    sequence = torch.roll(sequence, shifts=shift_idx.item(), dims=1)
        sequence = sequence.permute(1, 0, 2, 3)
        return sequence

class Audio_vision_dataset_v2(Dataset):  
    def __init__(self, llama2_dir, video_dir, audio_dir, label_path, sub_path, time_step, cfg):
        self.frame_rate = 24.0

        # ############# audio parameters #################
        self.resample = 16000
        self.n_mels = 64
        self.fixed_length = 500
        self.stats = [-5.4919195, 5.0389895]
        self.melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
        )
        self.normalizer = PrecomputedNorm(self.stats)

        # ############# vision parameters ################
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.img_size = 224
        self.num_frame = 16
        self.time_step = time_step

        self.transform_data = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x / 255.0),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        # ############# sub parameters ################
        self.tokenizer = AutoTokenizer.from_pretrained(llama2_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # ############# label file #######################
        self.classes_for_all_imgs = []
        self.annotation = []
        self.sub = []
        self.video = []
        self.audio = []
        
        self.annotation_1 = pd.read_pickle(label_path)
        self.sub_1 = pd.read_pickle(sub_path)
        self.video_1 = video_dir
        self.audio_1 = audio_dir

        for i in range(len(self.annotation_1)):
            a = [1, i, self.annotation_1[i][2]]
            self.annotation.append(a)
            sub = self.sub_1[i][-1]
            sub = re.sub("\n", " ", sub)
            sub = re.sub("-", " ", sub)
            b = [1, i, sub]
            self.sub.append(b)
            c = [1, i, self.video_1]
            self.video.append(c)
            d = [1, i, self.audio_1]
            self.audio.append(d)
            clas_id = self.annotation_1[i][2] 
            self.classes_for_all_imgs.append(clas_id)

        self.melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.resample, n_mels=self.n_mels
        )

    def __len__(self):
        return len(self.annotation)

    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs

    def __getitem__(self, index):
        # ########## sample ground truth label #####################
        label = self.annotation[index][2] 

        # ############## sample audio data #########################
        melspectogram_db = self.load_audio(index)

        # ############## sample video data #########################
        sequence = self.load_image(index, label)

        # ############## sample sub data #########################
        sub = self.sub[index][2]
        if sub == "":
            sub += " "
        sub_tokens = self.tokenizer.encode_plus(
            text=sub,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=64,
            pad_to_max_length=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = sub_tokens["input_ids"].squeeze(0)
        attention_mask = sub_tokens["attention_mask"].squeeze(0)

        return sequence, melspectogram_db, input_ids, attention_mask, label

    def load_audio(self, index):
        audio_path = self.audio[index][2]
        no = self.annotation[index][1]

        wav_name = osp.join(audio_path, str(no).zfill(5) + ".wav")
        soundData, sample_rate = torchaudio.load(wav_name)
        if soundData.shape[0] > 1:
            soundData = torch.mean(soundData, dim=0).unsqueeze(0)

        resampler = T.Resample(sample_rate, self.resample)
        soundData = resampler(soundData)
        frame_length = soundData.shape[1] / self.resample

        if frame_length > self.time_step:
            start_time = int((frame_length - self.time_step)/2.)
            end_time = (start_time + self.time_step) 
            soundData = soundData[:, start_time*self.resample:end_time*self.resample]
        else:
            tmp = torch.zeros(1, self.resample * self.time_step)
            tmp[:, : soundData.shape[1]] = soundData
            soundData = tmp

        melspectogram = self.normalizer(
            (
                self.melspectrogram_transform(soundData) + torch.finfo(torch.float).eps
            ).log()
        )

        return melspectogram  

    def load_image(self, index, label):
        video_path = self.video[index][2]
        no = self.annotation[index][1]

        path = osp.join(video_path, str(no).zfill(5) + ".mp4")

        video = torchvision.io.read_video(path)
        frame_rate = video[2]["video_fps"]
        len_time = int(float(video[0].shape[0]) / frame_rate)
        
        start_time = 0
        end_time = 0
        #=================== step frame_rate=========================
        if len_time > self.time_step:
            start_time = int((len_time - self.time_step) / 2 * frame_rate)
            end_time = start_time + int(self.time_step * frame_rate)
        else:
            start_time = 0
            end_time = int(len_time * frame_rate)

        video = video[0][start_time:end_time, :, :, :]

        # --------------------------------------------------------------------------------------------------------#
        sequence = torch.zeros((3, self.num_frame, self.img_size, self.img_size))

        step = int(float(video.shape[0]) / float(self.num_frame))
        if step >= 1:
            table = torch.arange(0, video.shape[0], step)
            for i in range(self.num_frame):
                idx = table[i]
                img = video[idx : idx + 1, :, :, :].squeeze(0).numpy().astype("uint8")
                img = Image.fromarray(img)
                img = self.transform_data(img).unsqueeze(1)
                sequence[:, i : i + 1, :, :] = img

        sequence = sequence.permute(1, 0, 2, 3)

        return sequence



