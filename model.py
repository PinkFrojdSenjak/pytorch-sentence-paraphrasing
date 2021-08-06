
from transformers import EncoderDecoderModel
import torch
from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

config_encoder = BertConfig()
config_decoder = BertConfig()
config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder,config_decoder)
model = EncoderDecoderModel(config=config)

config_encoder = model.config.encoder
config_decoder  = model.config.decoder

config_decoder.is_decoder = True
config_decoder.add_cross_attention = True
model.save_pretrained('my-model')
encoder_decoder_config = EncoderDecoderConfig.from_pretrained('my-model')
model = EncoderDecoderModel.from_pretrained('my-model', config=encoder_decoder_config)

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text = 'There is manuscript evidence that Austen continued to work on these pieces as late as the period 1809   11  and that her niece and nephew  Anna and James Edward Austen  made further additions as late as 1814'
encoding = tokenizer.encode_plus(text, add_special_tokens = True, 
    truncation = True, padding = "max_length", 
    return_attention_mask = True, return_tensors = "pt")

print(model(encoding))
