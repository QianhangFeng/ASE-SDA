import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from transformers import BartForConditionalGeneration, BartConfig, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput

from tools.optim_utils import get_optimizer
from warmup_scheduler import GradualWarmupScheduler

from models.audio_encoder_config import AudioEncoderConfig
from models.audio_encoder import AudioEncoderModel
from data_handling.aac_tokenizer import AACTokenizer
from .token_encoder import TokenEncoder, Projection

from tools.utils import decode_output
from eval_metrics import evaluate_metrics


token_path = "/home/feng/desktop/mutiset/WavCaps-master/captioning/settings/tokenizer.json"

class BartCaptionModel(pl.LightningModule):

    def __init__(self, config):
        super(BartCaptionModel, self).__init__()

        self.config = config
        # encoder
        encoder_config = AudioEncoderConfig(**config["audio_encoder_args"],
                                            audio_args=config["audio_args"])
        self.encoder = AudioEncoderModel(encoder_config)
        

        self.freeze_token_encoder = False
        self.mix_audio = True
        
        self.freeze_decoder = False
        
        
        dropout_rate = 0.5

        self.freeze_encoder = True
        self.decoder_pretrained = False
        self.pos_enhance = True

        self.token_proj = Projection(2*config['token_encoder_args']['hidden_size'], config['text_decoder_args']['hidden_size'], dropout_rate=dropout_rate)
        self.token_encoder = TokenEncoder(config)

        # bart decoder
        decoder_name = config["text_decoder_args"]["name"]

        if self.decoder_pretrained:
            print("Use PreTrained BART")
            self.tokenizer = BartTokenizer.from_pretrained(decoder_name)
            self.vocab_size = len(self.tokenizer)
            self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        else:
            print("Use Transformer")
            self.tokenizer = AACTokenizer().from_file(token_path)
            self.vocab_size = self.tokenizer.get_vocab_size()
            self.decoder = BartForConditionalGeneration(self.config_Small(decoder_name)) # Samll, Base
        
        self.decoder.model.encoder = None

        if self.freeze_encoder:
            freeze(self.encoder)

        if self.freeze_token_encoder:
            freeze(self.token_encoder)
            
        if self.freeze_decoder:
            freeze(self.decoder)
            # freeze(self.token_proj)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.2)
        
        self.validation_step_outputs = []

    def forward_encoder(self, audios):
        if self.freeze_encoder:
            self.encoder.eval()
        outputs = self.encoder(audios).last_hidden_state["frame_embeddings"]
        encoder_outputs = torch.mean(outputs, dim=1).squeeze()
        if encoder_outputs.shape[1] != 94:
            encoder_outputs = F.interpolate(encoder_outputs.permute(0, 2, 1), size=94, mode='linear').permute(0, 2, 1)
        return encoder_outputs

    def forward_token_encoder(self, encoder_outputs, training=False):
        if self.freeze_token_encoder:
            self.token_encoder.eval()
        encoder_outputs, mixdict = self.token_encoder(encoder_outputs, training=self.mix_audio if training else False)
        return encoder_outputs, mixdict

    def forward_decoder(self, text, encoder_outputs):
        if self.decoder_pretrained:
            text_tokens = self.tokenizer(text,
                        padding='longest',
                        truncation=True,
                        max_length=30,
                        return_tensors="pt")   
            input_ids = text_tokens["input_ids"].to(self.device)
            attention_mask = text_tokens["attention_mask"].to(self.device)
        else:
            input_ids = []
            attention_mask = []
            res = self.tokenizer.encode_batch(text)
            for text_token in res:
                input_ids.append(text_token.ids)
                attention_mask.append(text_token.attention_mask)

            input_ids=torch.tensor(input_ids).to(self.device)
            attention_mask=torch.tensor(attention_mask).to(self.device)

        decoder_targets = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        decoder_input_ids = shift_tokens_right(
            decoder_targets, self.decoder.config.pad_token_id, self.decoder.config.decoder_start_token_id
        )

        if self.freeze_decoder:
            self.decoder.eval()
        d = self.decoder.model.decoder
        decoder_inputs_embeds = d.embed_tokens(decoder_input_ids) * d.embed_scale
        
        decoder_outputs = self.decoder(
            decoder_attention_mask=attention_mask,
            decoder_inputs_embeds=decoder_inputs_embeds,
            encoder_outputs=(encoder_outputs,),
            return_dict=True,
            pos_enhance=self.pos_enhance
        )
        
        lm_logits = decoder_outputs["logits"]

        loss = self.loss_fct(lm_logits.view(-1, self.vocab_size), decoder_targets.view(-1))
            
        return loss

    def forward(self, audios, specs, captions):
        audio_embs = self.forward_encoder(audios)
        if audio_embs.shape[1] != 94:
                audio_embs = F.interpolate(audio_embs.permute(0, 2, 1), size=94, mode='linear').permute(0, 2, 1)
        

        token_encoder_emb, _ = self.forward_token_encoder(encoder_outputs=audio_embs, training=True) # 2 proj
        token_encoder_emb = torch.cat((token_encoder_emb, audio_embs), dim=2)

        token_encoder_emb, _ = self.token_proj(token_encoder_emb, specs=specs, mix_up=False)

        # token_encoder_emb, _ = self.token_proj(token_encoder_emb, specs=None, mix_up=False)
        # token_encoder_emb = specs + token_encoder_emb

        loss = self.forward_decoder(captions, token_encoder_emb)
        return loss

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        audios, specs, captions, names = batch
        if audios is None or captions is None:
            return
        loss = self.forward(audios, specs, captions)
        return loss

   
    def validation_step(self, batch, batch_idx):
        audios, gt_captions, names = batch
        pred_captions = []
        
        for beamsize in range(2, 5):
            output = self.generate(audios=audios, num_beams=beamsize)
            pred_captions.append(output)

        res = {'pred_captions': pred_captions, 'gt_captions': gt_captions, 'names': names}
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs

        pred_captions_2 = []
        pred_captions_3 = []
        pred_captions_4 = []
        gt_captions = []
        names = []

        for output in outputs:
            pred_captions_2.extend(output['pred_captions'][0])
            pred_captions_3.extend(output['pred_captions'][1])
            pred_captions_4.extend(output['pred_captions'][2])
            gt_captions.extend(output['gt_captions'])
            names.extend(output['names'])
        
        captions_pred_2, captions_gt = decode_output(pred_captions_2, gt_captions, names)
        captions_pred_3, captions_gt = decode_output(pred_captions_3, gt_captions, names)
        captions_pred_4, captions_gt = decode_output(pred_captions_4, gt_captions, names)

        captions_preds = [captions_pred_2, captions_pred_3, captions_pred_4]

        metrics = [evaluate_metrics(captions_pred, captions_gt) for captions_pred in captions_preds]
        spiders = [metric['spider']['score'] for metric in metrics]
        spider = max(spiders)

        self.log("spider", value=spider, sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters(),
                            lr=self.config['optim_args']['lr'],
                            betas=self.config['optim_args']['betas'],
                            eps=self.config['optim_args']['eps'],
                            momentum=self.config['optim_args']['momentum'],
                            weight_decay=self.config['optim_args']['weight_decay'],
                            optimizer_name=self.config['optim_args']['optimizer_name'])

        scheduler_temp = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_temp)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler_warmup}

    def generate(self,
                 audios,
                 use_nucleus_sampling=False,
                 num_beams=3,
                 max_length=30,
                 min_length=2,
                 top_p=0.9,
                 repetition_penalty=1.0,
                 ):
        # noise
        audio_embs = self.forward_encoder(audios)
        token_encoder_emb, _ = self.forward_token_encoder(audio_embs, training=False) # 2 proj
        encoder_outputs = torch.cat((token_encoder_emb, audio_embs), dim=2)
        # proj
        encoder_outputs, _ = self.token_proj(encoder_outputs, specs=None, mix_up=False)


        decoder_input_ids = torch.zeros((encoder_outputs.size(0), 1)).long().to(self.device)
        decoder_input_ids[:, 0] = self.decoder.config.decoder_start_token_id
        decoder_attention_mask = torch.ones((encoder_outputs.size(0), 1)).long().to(self.device)

        d = self.decoder.model.decoder
        decoder_inputs_embeds = d.embed_tokens(decoder_input_ids) * d.embed_scale

        encoder_outputs = BaseModelOutput(encoder_outputs, None, None)

        if use_nucleus_sampling:
            outputs = self.decoder.generate(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=None,
                pos_enhance=self.pos_enhance,
                decoder_inputs_embeds=decoder_inputs_embeds,
                decoder_attention_mask=decoder_attention_mask,
                encoder_outputs=encoder_outputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=True,
                top_p=top_p,
                num_return_sequences=1,
                repetition_penalty=1.1)
        else:
            outputs = self.decoder.generate(input_ids=None,
                                            attention_mask=None,
                                            decoder_input_ids=None,
                                            pos_enhance=self.pos_enhance,
                                            decoder_attention_mask=decoder_attention_mask,
                                            encoder_outputs=encoder_outputs,
                                            head_mask=None,
                                            decoder_head_mask=None,
                                            inputs_embeds=None,
                                            decoder_inputs_embeds=decoder_inputs_embeds,
                                            use_cache=None,
                                            output_attentions=None,
                                            output_hidden_states=None,
                                            max_length=max_length,
                                            min_length=min_length,
                                            num_beams=num_beams,
                                            repetition_penalty=repetition_penalty)
        if self.decoder_pretrained:  
            captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            outputs=outputs.cpu().numpy().tolist()      
            captions = self.tokenizer.decode_batch(outputs)

        return captions
     
    def config_Small(self, decoder_name):
        bart_config = BartConfig.from_pretrained(decoder_name)

        bart_config.max_position_embeddings=self.config['text_decoder_args']['hidden_size']
        bart_config.decoder_ffn_dim=2048
        bart_config.encoder_ffn_dim=2048
        bart_config.decoder_attention_heads=8 # 8
        bart_config.encoder_attention_heads=8 # 8
        bart_config.d_model=self.config['text_decoder_args']['hidden_size'] # 768
        
        bart_config.decoder_layers=6
        bart_config.encoder_layers=6

        bart_config.dropout=0.2
        bart_config.num_beams=4 # 3
        bart_config.vocab_size=self.vocab_size
        bart_config.bos_token_id = self.tokenizer.bos_token_id
        bart_config.pad_token_id = self.tokenizer.pad_token_id
        bart_config.eos_token_id = self.tokenizer.eos_token_id

        return bart_config
      
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

def freeze(item):
    for name, param in item.named_parameters():
        param.requires_grad = False