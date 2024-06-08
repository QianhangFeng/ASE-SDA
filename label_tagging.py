#!/usr/bin/env python3
# coding: utf-8
# @Author  : Xinhao Mei @CVSSP, University of Surrey
# @E-mail  : x.mei@surrey.ac.uk


import os
import argparse
import torch
import time
import wandb
from pprint import PrettyPrinter
import torch
import platform
import argparse
from ruamel.yaml import YAML
from loguru import logger
from warmup_scheduler import GradualWarmupScheduler
from data_handling.datamodule import AudioCaptionDataModule
# model
from models.bart_captioning import BartCaptionModel
from models.bert_captioning import BertCaptionModel

from tools.optim_utils import get_optimizer, cosine_lr, step_lr
from tools.utils import setup_seed, set_logger

from tools.schedulers import CosDecayScheduler

# validate
from pretrain import validate as val_fun, train 
from new_pretrain import validate as new_val_fun

from tools.schedulers import CosDecayScheduler

from tqdm import tqdm

from models.convnext.src.audioset_convnext_inf.utils.utilities import read_audioset_label_tags

import numpy as np
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Settings.")
    parser.add_argument("-c", "--config", default="settings/settings.yaml", type=str,
                        help="Name of the setting file.")
    
    # parser.add_argument("-n", "--name", default="auto", type=str,
    #                 help="Name of the setting file.")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml = YAML(typ="safe", pure=True)
        config = yaml.load(f)

    mode = int(config["exp_name"][-1])

    print(mode)
    # assert(config["mod"] in ["state_1", "state_2", "fintune"], "No such mod!!!")

    setup_seed(config["seed"])

    exp_name = config["exp_name"]
    # exp_name = args.name

    folder_name = "{}_lr_{}_batch_{}_seed_{}".format(exp_name,
                                                     config["optim_args"]["lr"],
                                                     config["data_args"]["batch_size"],
                                                     config["seed"])

    model_output_dir, log_output_dir = set_logger(folder_name)

    main_logger = logger.bind(indent=1)

    # set up model
    device, device_name = (config["device"],
                           torch.cuda.get_device_name(torch.cuda.current_device())) \
        if torch.cuda.is_available() else ("cpu", platform.processor())
    main_logger.info(f"Process on {device_name}, at {device}")

    # data loading
    datamodule = AudioCaptionDataModule(config, config["data_args"]["dataset"])
    train_loader = datamodule.train_dataloader(is_distributed=False)
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    model = BartCaptionModel(config).to(device=device)
    encoder = model.encoder

    model.eval()

    audio_list = []



    for i, batch_data in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        audios, caption_dict, audio_names, audio_ids = batch_data
        tmp_dict = {"audio_name": audio_names[0]}
        audios = audios.to(device)
        output = encoder(audios)
        probs = output["clipwise_output"]
        lb_to_ix, ix_to_lb, id_to_ix, ix_to_id = read_audioset_label_tags("/home/feng/desktop/WavCaps-master/captioning/outputs/val/class_labels_indices.csv")
        threshold = 0.25
        sample_labels = np.where(probs[0].clone().detach().cpu() > threshold)[0]

        sample_dict = {}
        for i, l in enumerate(sample_labels):
            sample_dict[i] = {"label": ix_to_lb[l], "prob": probs[0, l].item()}
        tmp_dict["sample_dict"] = sample_dict
        audio_list.append(tmp_dict)

    
    with open("/home/feng/desktop/WavCaps-master/captioning/outputs/val/test_beam_3.json", "r") as fcc_file:
        meta = json.load(fcc_file)

        names_meta = [data["audio_name"] for data in meta]
        names_label = [data["audio_name"] for data in audio_list]

        for i, name in enumerate(names_label):
            index = names_meta.index(name)
            index_m = meta[index]["index"]
            for k in audio_list[i].keys():
                tmp = {}
                if k != "audio_name":
                    tmp[k] = audio_list[i][k]
            meta[index]["pred_labels"] = tmp
        
        with open(os.path.join("/home/feng/desktop/WavCaps-master/captioning/outputs/val", "test_meta_beam_3.json"),"w") as f:
            json.dump(meta,f)


if __name__ == "__main__":
    main()
