#!/usr/bin/env python
from transformers import RobertaConfig

config = RobertaConfig.from_pretrained("roberta-large")
config.save_pretrained("./configs/large")

config = RobertaConfig.from_pretrained("roberta-base")
config.save_pretrained("./configs/base")
