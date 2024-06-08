from ruamel.yaml import YAML
from data_handling.captionmodule import Data_iter # use csv, diff
import torch
from models.new_bart_token_enc import BartCaptionModel # 无 info_nce

from tools.utils import setup_seed
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

model_path = "/home/feng/desktop/WavCaps-master/captioning/outputs/backup/fusion_cat2_Small_cl_new_4_lr_8e-06_batch_8_seed_42/models/best_model_spider.pt"
checkpoint_path = ''



def main():

    with open('settings/settings.yaml', 'r') as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)

    folder_name = '{}_lr_{}_batch_{}_seed_{}'.format(config['exp_name'],
                                            config['optim_args']['lr'],
                                            config['data_args']['batch_size'],
                                            config['seed'])

    log_output_dir = Path('outputs', folder_name, 'logging')
    model_output_dir = Path('outputs', folder_name, 'models')
    log_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(config['seed'])

    data_iter = Data_iter(config)
    train_loader = data_iter.train_dataloader()
    val_loader = data_iter.val_dataloader()

    config['optim_args']['data_size'] = len(train_loader)

    if checkpoint_path != '':
        model = BartCaptionModel.load_from_checkpoint(checkpoint_path=checkpoint_path, config=config)
    else:
        model = BartCaptionModel(config)

        if model_path != '':
            checkpoint = torch.load(model_path)['model']
            model_dict = model.state_dict()
            for k in model_dict.keys():
                model_dict[k] = checkpoint[k]
            model.load_state_dict(model_dict)


    checkpoint_callback = ModelCheckpoint(
        dirpath=model_output_dir,
        filename='{epoch:02d}-{spider:.4f}',
        monitor='spider',
        save_top_k=2,
        mode='max',
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        verbose=True
    )

    earlyStopping_callback = EarlyStopping(
        monitor="spider", 
        min_delta=0.00, 
        patience=10, 
        verbose=False, 
        mode="max"
    )

    logger = CSVLogger(log_output_dir, name=config['exp_name'])

    trainer = pl.Trainer(max_epochs=50, devices=[0, 1, 2, 3, 4], num_sanity_val_steps=0, callbacks=[checkpoint_callback, earlyStopping_callback], logger=logger)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

test_path = "/home/feng/desktop/WavCaps-master/captioning/outputs/backup/fusion_cat2_Small_cl_new_4_lr_8e-06_batch_8_seed_42/models/best_model_spider.pt"
# test_path = ''
test_checkpoint_path = ''
# test_checkpoint_path = '/home/feng/desktop/mutiset/WavCaps-master/captioning/outputs/mutiset_clotho_or_3_lr_4e-06_batch_8_seed_42/models/epoch=00-spider=0.3242.ckpt'
# test_checkpoint_path = '/home/feng/desktop/mutiset/WavCaps-master/captioning/outputs/mutiset_clotho_or_3_lr_4e-06_batch_8_seed_42/models/epoch=01-spider=0.3203.ckpt'
# test_checkpoint_path = '/home/feng/desktop/mutiset/WavCaps-master/captioning/outputs/backup1/mutiset_clotho_3_lr_1e-05_batch_64_seed_7588/models/epoch=03-spider=0.3392.ckpt'

def test():
    with open('settings/settings.yaml', 'r') as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)

    folder_name = '{}_lr_{}_batch_{}_seed_{}'.format(config['exp_name'],
                                            config['optim_args']['lr'],
                                            config['data_args']['batch_size'],
                                            config['seed'])

    log_output_dir = Path('outputs', folder_name, 'test_logging')
    log_output_dir.mkdir(parents=True, exist_ok=True)

    setup_seed(config['seed'])

    data_iter = Data_iter(config)
    val_loader = data_iter.val_dataloader()

    model = BartCaptionModel(config)

    if test_path != "":
        checkpoint = torch.load(test_path)['model']
        model_dict = model.state_dict()
        for k in model_dict.keys():
            model_dict[k] = checkpoint[k]
        model.load_state_dict(model_dict)

    test_logger = CSVLogger(log_output_dir, name=config['exp_name'])

    if test_checkpoint_path == '':
        trainer = pl.Trainer(devices=[0], num_sanity_val_steps=0, logger=test_logger)
    else:
        trainer = pl.Trainer(checkpoint_path=test_checkpoint_path, devices=0, num_sanity_val_steps=0, logger=test_logger)

    trainer.validate(model, val_loader)


if __name__ == '__main__':
    # main()
    test()

