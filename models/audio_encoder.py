import torch
from transformers.modeling_outputs import BaseModelOutput

from transformers import PreTrainedModel
from models.audio_encoder_config import AudioEncoderConfig

# convnext
from models.convnext.src.audioset_convnext_inf.pytorch.convnext import ConvNeXt, convnext_tiny

class AudioEncoderModel(PreTrainedModel):
    config_class = AudioEncoderConfig

    def __init__(self, config):
        super(AudioEncoderModel, self).__init__(config)
        self.config = config

        model_path = '/home/feng/desktop/WavCaps-master/captioning/pretrained_models/audio_enocder/convnext_tiny_465mAP_BL_AC_70kit.pth'
        # self.audio_enc =  ConvNeXt.from_pretrained(pretrained_checkpoint_path=model_path)
        self.audio_enc =  convnext_tiny(
                            pretrained=False,
                            strict=False,
                            drop_path_rate=0.0,
                            after_stem_dim=[252, 56],
                            use_speed_perturb=False,
                        )
        checkpoint = torch.load(model_path)
        self.audio_enc.load_state_dict(checkpoint['model'])


        if config.freeze:
            for name, param in self.audio_enc.named_parameters():
                if "fc1" not in name:
                    param.requires_grad = False

    def forward(self, input_ids,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
                ):

        audio_embeds = self.audio_enc(input_ids)
        if not return_dict:
            return (audio_embeds, )

        return BaseModelOutput(audio_embeds, None, None)
    


