import torch
from data_handling.caption_dataset import AudioCaptionDataset, resample
from torch.utils.data import DistributedSampler, DataLoader
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import librosa

import os
from torch.utils.data import Dataset
from data_handling.text_transform import text_preprocess
import json

class Data_iter(pl.LightningDataModule):
    def __init__(self, config: dict):
        super(Data_iter, self).__init__()

        self.config = config

        self.fn_train=config['data_args']['fn_train']

        self.batch_size = config["data_args"]["batch_size"]
        self.num_workers = config["data_args"]["num_workers"]

    def train_dataloader(self):
        dataset = Total_dataset(self.config, self.fn_train)
        return self.get_train_dataloader(dataset)
  
    def val_dataloader(self):
        dataset = AudioCaptionDataset(self.config["audio_args"], 'Clotho', split='test')
        return self.get_test_dataloader(dataset)
    
    def test_dataloader(self):
        dataset = AudioCaptionDataset(self.config["audio_args"], 'Clotho', split='test')
        return self.get_test_dataloader(dataset)

    def get_train_dataloader(self, data_set, is_distributed=False, num_tasks=0, global_rank=0):
        sampler = _get_sampler(
            dataset=data_set,
            shuffle=True,
            is_distributed=is_distributed,
            num_tasks=num_tasks,
            global_rank=global_rank)
        shuffle = sampler is None

        return DataLoader(
            dataset=data_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=train_collate_fn,
            drop_last=True
        )


    def get_test_dataloader(self, data_set):
        return DataLoader(data_set,
                          batch_size=64,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          sampler=None,
                          shuffle=False,
                          collate_fn=val_collate_fn,
                          drop_last=False
        )

add_ac = False
add_wc = False

useSpec = False # 2个

class Total_dataset(Dataset):
    def __init__(self, config: dict, fn_train: bool = False):
        super(Total_dataset, self).__init__()

        self.train_dataset_list = []
        self.expect_sr = config["audio_args"]['sr']

        if fn_train:
            if add_ac:
                self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                        'AudioCaps',
                                                        split='train'))
                self.train_dataset_list.append(AudioCaptionDataset(config['audio_args'],
                                                        'AudioCaps',
                                                        split='val'))
                self.train_dataset_list.append(AudioCaptionDataset(config['audio_args'],
                                                        'AudioCaps',
                                                        split='test'))
            if add_wc:
                self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                        'WavCaps',
                                                        sub_set='AudioSet',
                                                        split='train'))
                self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                        'WavCaps',
                                                        sub_set='BBC',
                                                        split='train'))
                self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                        'WavCaps',
                                                        sub_set='FreeSound',
                                                        split='train'))
                self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                        'WavCaps',
                                                        sub_set='SoundBible',
                                                        split='train'))
            
        self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                'Clotho',
                                                split='train'))
        self.train_dataset_list.append(AudioCaptionDataset(config["audio_args"],
                                                'Clotho',
                                                split='val'))
        

        self.wav_paths = []
        self.captions = []

        self.blacklist = set()
        with open('/home/feng/desktop/mutiset/WavCaps-master/captioning/settings/blacklist.json', 'r') as file:
            self.blacklist = set(json.load(file))
        
        count = 0
        for dataset in self.train_dataset_list:
            wavpaths = dataset.wav_paths
            captions = dataset.captions
            for wavpath, caption in zip(wavpaths, captions):
                if wavpath in self.blacklist:
                    count += 1
                    continue
                self.wav_paths.append(wavpath)
                self.captions.append(caption)

        print(f'Data in blacklist: {count}')
        
       
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

def train_collate_fn(batch):
    if isinstance(batch, list):
        batch = [(audio, caption, audio_name, lens) for (audio, caption, audio_name, lens) in
                 batch if audio is not None]
    if not batch:
        return None, None, None, None

    audios = [i[0] for i in batch]
    lens = [i[-1] for i in batch]
    audio_list = pad(audios, lens)
    caption_list = [i[1] for i in batch]
    audio_name_list = [i[2] for i in batch]

    audio_arr = np.array(audio_list)

    if useSpec:
        specs = librosa.feature.melspectrogram(y=audio_arr, sr = 32000, n_fft=1024, n_mels = 64, hop_length=320, fmax=14000, fmin=50)
        specs = torch.from_numpy(specs)
        if specs.shape[2] != 768:
            specs = F.interpolate(specs, size=768, mode='linear')
        if specs.shape[1] != 94:
            specs = F.interpolate(specs.permute(0, 2, 1), size=94, mode='linear').permute(0, 2, 1)
    else:
        specs = None
    
    audio_arr = torch.from_numpy(audio_arr)

    return audio_arr, specs, caption_list, audio_name_list

def val_collate_fn(batch):
    audios = [i[0] for i in batch]
    lens = [i[-1] for i in batch]
    audio_list = pad(audios, lens)
    caption_list = [i[1] for i in batch]
    audio_name_list = [i[2] for i in batch]

    audio_arr = torch.from_numpy(np.array(audio_list))

    return audio_arr, caption_list, audio_name_list

def _get_sampler(dataset,
                 shuffle,
                 is_distributed,
                 num_tasks,
                 global_rank):
    # do not return a sampler if is not in distributed mode
    # a default RandomSampler is used in this case
    if not is_distributed:
        return None

    return DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle
    )

def pad(audios, lens):
    max_len = max(lens)
    final_audios = []
    for i, (audio, audio_len) in enumerate(zip(audios, lens)):
        audio = torch.tensor(audio)
        if audio_len<max_len:
            pad_length = max_len - audio_len
            audio = F.pad(audio, [0, pad_length], "constant", 0.0)
        final_audios.append(audio)
    waveforms = torch.stack(final_audios, dim=0)
    return waveforms
