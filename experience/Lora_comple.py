# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:02:57 2025

@author: 17808
"""

from peft import LoraConfig, get_peft_model

# 仅修改注意力层的输出投影
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(base_model, lora_config)  # <1%参数可训练