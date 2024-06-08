import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from torch.distributions.beta import Beta
from typing import Any, Dict, List, Union, Iterable

from transformers import BartForConditionalGeneration, BartConfig

class TokenEncoder(nn.Module):

    def __init__(self, config):
        super(TokenEncoder, self).__init__()

        self.config = config

        self.hidden_size = config['token_encoder_args']['hidden_size']

        bart_config = self.config_Base()
        
        self.token_encoder = BartForConditionalGeneration(bart_config).model.encoder
        self.token_encoder.embed_tokens = None

        
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config['token_encoder_args']['seq_len'], self.hidden_size)
        )

        self.query_proj = Projection(2*768, config['token_encoder_args']['hidden_size'])

    @property
    def device(self):
        return list(self.parameters())[0].device

    def forward(self, encoder_outputs, training=True):
        repeated_query_tokens = self.query_tokens.repeat(encoder_outputs.shape[0], 1, 1)
        encoder_outputs = torch.cat((repeated_query_tokens, encoder_outputs), dim=2)

        encoder_outputs, mixdict = self.query_proj(encoder_outputs, mix_up=training)

        encoder_outputs = self.token_encoder(
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=encoder_outputs,
            output_attentions=None,
            output_hidden_states=None,
            pos_enhance=True,
            return_dict=True
        )
        return encoder_outputs.last_hidden_state, mixdict
    
    
    def config_Base(self):
        bart_config = BartConfig.from_pretrained(self.config["text_decoder_args"]["name"])
        return bart_config
    

class Projection(torch.nn.Module):
    def __init__(self, idim, odim, dropout_rate=0.5):
        super(Projection, self).__init__()

        self.proj = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(idim, odim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )

    def forward(self, x, specs=None, mix_up=True):
        # bsz, c, t, d_m
        mixup_dict = None
        if mix_up:
            x, mixup_dict = randperm_audio(x)
        if specs is not None:
            specs = torch.cat((specs, specs), dim=2)
            x = x + specs

        x=self.proj(x)
        return x, mixup_dict

def randperm_audio(x):
    lbd = sample_lambda(alpha = 0.4, asymmetric=True,size=())
    
    indexes = randperm_diff(x.shape[0], x.device)
    x = x * lbd + x[indexes] * (1.0 - lbd)

    mixup_dict = {"indexes": indexes, "lbd": lbd}
    return x, mixup_dict

def randperm_diff(size: int,
    device: Union[str, torch.device, None],
    generator: Union[None, int, torch.Generator] = None,
) -> Tensor:
    if size < 2:
        raise ValueError(f"Invalid argument {size=} < 2 for randperm_diff.")

    if isinstance(generator, int):
        generator = torch.Generator().manual_seed(generator)

    perm_kws: Dict[str, Any] = dict(generator=generator, device=device)
    arange = torch.arange(size, device=device)
    perm = torch.randperm(size, **perm_kws)

    while perm.eq(arange).any():
        perm = torch.randperm(size, **perm_kws)
    return perm

def sample_lambda(alpha: float | Tensor,
    asymmetric: bool, size: Iterable[int] = (),
) -> Tensor:
    """
    :param alpha: alpha hp to control the Beta distribution.
        Values closes to 0 means distribution will peak up at 0 and 1, while values closes to 1 means sampling from an uniform distribution.
    :param asymmetric: If True, lbd value will always be in [0.5, 1], with values close to 1.
    :param size: The size of the sampled lambda(s) value(s). defaults to ().
    :returns: Sampled values of shape defined by size argument.
    """
    tensor_kwds: dict[str, Any] = dict(dtype=torch.get_default_dtype())
    size = torch.Size(size)

    if alpha == 0.0:
        if asymmetric:
            return torch.full(size, 1.0, **tensor_kwds)
        else:
            return torch.rand(size).ge(0.5).to(**tensor_kwds)

    alpha = torch.as_tensor(alpha, **tensor_kwds)
    beta = Beta(alpha, alpha)
    lbd = beta.sample(size)
    if asymmetric:
        lbd = torch.max(lbd, 1.0 - lbd)
    lbd = max(lbd, 1.0 - lbd)
    return lbd

