from peft import LoraConfig, TaskType

def get_lora_config(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    use_rslora=True,
    target_modules=None,
    use_qlora=False
):
    """
    Get LoRA configuration for Qwen2.5-VL
    
    Args:
        r (int): LoRA rank
        lora_alpha (int): LoRA alpha parameter
        lora_dropout (float): LoRA dropout rate
        use_rslora (bool): Whether to use rank stabilized LoRA
        target_modules (list): Target modules for LoRA adaptation
        use_qlora (bool): Whether to use QLoRA (4-bit quantization)
    """
    
    if target_modules is None:
        # Focus on language model attention layers
        target_modules = [
            "q_proj",
            "v_proj",
            "k_proj", 
            "o_proj",
            # Uncomment below for MLP layers (uses more memory)
            # "gate_proj",
            # "up_proj", 
            # "down_proj",
        ]
    
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
    )

def get_qlora_config():
    """Get QLoRA configuration for memory efficiency"""
    from transformers import BitsAndBytesConfig
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )