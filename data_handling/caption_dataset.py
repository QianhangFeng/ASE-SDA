import random
import pandas as pd
import librosa
import numpy as np
import os
import torchaudio
from torch.utils.data import Dataset
from data_handling.text_transform import text_preprocess
import json


def resample(audio_path, expect_sr):
    orig_sr = librosa.get_samplerate(audio_path)
    orig_time = librosa.get_duration(path=audio_path)

    segment_seconds = 30

    if orig_time < 5:
        raise ValueError

    if segment_seconds < orig_time:
        segment_start_time = random.randint(0, int(orig_time - segment_seconds))
        orig_seg_start_sample = round(segment_start_time * orig_sr)
        orig_seg_samples = round(segment_seconds * orig_sr)
    else:
        orig_seg_start_sample = 0
        orig_seg_samples = -1

    try:
        audio, fs = torchaudio.load(
            audio_path,
            frame_offset=orig_seg_start_sample,
            num_frames=orig_seg_samples
        )
    except RuntimeError:
        raise ValueError

    # (channel, time)
    audio = audio[0, :]

    if orig_sr != expect_sr:
        audio = librosa.resample(audio.numpy(), orig_sr=orig_sr, target_sr=expect_sr, res_type="fft")
        
    return audio.tolist()


def pad_or_truncate(x, audio_length):
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        max_start = len(x) - audio_length
        start = random.randint(0, max_start)
        x = x[start: start + audio_length]
        return np.array(x)


def truncate(x, max_length):
    max_start = len(x) - max_length
    start = random.randint(0, max_start)
    x = x[start: start + max_length]
    return np.array(x)


def read_json(json_path):
    with open(json_path, 'r') as json_file:
        json_dict = json.load(json_file)
        return json_dict


class AudioCaptionDataset(Dataset):

    def __init__(self,
                 audio_config: dict,
                 dataset: str = 'AudioCaps',
                 sub_set="",
                 split: str = 'train', # val
                 ):
        super(AudioCaptionDataset, self).__init__()

        pd.set_option('mode.chained_assignment', None)

        self.dataset = dataset
        self.split = split
        self.expect_sr = audio_config['sr']


        if self.split == 'train':
            if self.dataset == 'AudioCaps':
                self.wav_path = audio_config['train_audiocaps_wav_path']
                self.cache = pd.read_csv(audio_config['train_audiocaps_csv_path'])
            elif self.dataset == 'WavCaps':
                self.wav_path = audio_config['train_wavcaps_wav_path']
                if sub_set == 'AudioSet':
                    self.cache = read_json(audio_config['train_wavcaps_json_path_as'])
                elif sub_set == 'BBC':
                    self.cache = read_json(audio_config['train_wavcaps_json_path_bbc'])
                elif sub_set == 'FreeSound':
                    self.cache = read_json(audio_config['train_wavcaps_json_path_fsd'])
                else:  # SoundBible
                    self.cache = read_json(audio_config['train_wavcaps_json_path_sb'])
            else:
                self.wav_path = audio_config['train_clotho_wav_path']
                self.cache = pd.read_csv(audio_config['train_clotho_csv_path'])
        elif self.split == 'val':
            if self.dataset == 'AudioCaps':
                self.wav_path = audio_config['val_audiocaps_wav_path']
                self.cache = pd.read_csv(audio_config['val_audiocaps_csv_path'])
            else:
                self.wav_path = audio_config['val_clotho_wav_path']
                self.cache = pd.read_csv(audio_config['val_clotho_csv_path'])
        else:
            if self.dataset == 'AudioCaps':
                self.wav_path = audio_config['test_audiocaps_wav_path']
                self.cache = pd.read_csv(audio_config['test_audiocaps_csv_path'])
            else:
                self.wav_path = audio_config['test_clotho_wav_path']
                self.cache = pd.read_csv(audio_config['test_clotho_csv_path'])

        if self.dataset == 'AudioCaps':
            self.num_captions_per_audio = 1
            self.captions = self.cache['caption'][:]
            self.wav_paths = [os.path.join(self.wav_path, '{}.wav'.format(audio_id)) for audio_id in
                              self.cache['audiocap_id']]

        elif self.dataset == 'WavCaps':
            self.num_captions_per_audio = 1
            self.captions = [data_dict['caption'] for data_dict in self.cache['data']]
            self.wav_paths = [
                os.path.join(self.wav_path, sub_set, '{}.flac'.format(data['id'].split('.')[0]))
                for data in self.cache['data']]
        # Clotho
        else:
            self.num_captions_per_audio = 5
            self.wav_paths = [os.path.join(self.wav_path, audio_id) for audio_id in self.cache['file_name']]
            self.captions = []
            for j in range(len(self.wav_paths)):
                tmp = []
                for i in range(0, self.num_captions_per_audio):
                    tmp.append(self.cache['caption_{}'.format(i + 1)][j])
                if self.split == 'train' or self.split == 'val':
                    self.captions += tmp
                else:
                    self.captions.append(tmp)
            
            if self.split == 'train' or self.split == 'val':
                wav_tmp_paths = []
                for j in range(len(self.wav_paths)):
                    for i in range(0, self.num_captions_per_audio):
                        wav_tmp_paths.append(self.wav_paths[j])
                self.wav_paths = wav_tmp_paths

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]

        if os.path.exists(wav_path) is False:
            return None, None, None, None
        
        try:
            audio = resample(wav_path, self.expect_sr)
        except ValueError:
            # 当文件损坏时会产生ValueError异常
            return None, None, None, None
        

        if isinstance(self.captions[index], list):
            # 数据清洗, 未添加标签
            caption = [text_preprocess(cap) for cap in self.captions[index]]
        else:
            caption = text_preprocess(self.captions[index])

        audio_name = wav_path.split("/")[-1]

        return audio, caption, audio_name, len(audio)
