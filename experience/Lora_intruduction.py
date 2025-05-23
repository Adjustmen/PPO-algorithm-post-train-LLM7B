from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,  # 低秩维度
    lora_alpha=32,  # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 仅微调注意力层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 应显示约0.1%参数可训练