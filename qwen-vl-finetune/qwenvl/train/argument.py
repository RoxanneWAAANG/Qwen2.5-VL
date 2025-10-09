import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

    # LoRA Configuration
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for fine-tuning"}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA (4-bit quantization with LoRA)"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoRA alpha parameter"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of target modules for LoRA. If None, use default modules."}
    )
    use_rslora: bool = field(
        default=True,
        metadata={"help": "Whether to use Rank-Stabilized LoRA"}
    )
    lora_modules_to_save: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated list of modules to save in addition to LoRA parameters"}
    )


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


def get_lora_target_modules(target_modules_str: Optional[str] = None) -> List[str]:
    """
    Get LoRA target modules for Qwen2.5-VL
    
    Args:
        target_modules_str: Comma-separated string of target modules
        
    Returns:
        List of target module names
    """
    if target_modules_str:
        return [module.strip() for module in target_modules_str.split(",")]
    
    # Default target modules for Qwen2.5-VL language model components
    return [
        "q_proj",
        "v_proj", 
        "k_proj",
        "o_proj",
        # Uncomment below for MLP layers (requires more memory)
        # "gate_proj",
        # "up_proj",
        # "down_proj",
    ]


def get_qlora_config():
    """Get QLoRA quantization configuration"""
    from transformers import BitsAndBytesConfig
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )


def get_lora_config(model_args: ModelArguments):
    """
    Get LoRA configuration from ModelArguments
    
    Args:
        model_args: ModelArguments instance
        
    Returns:
        LoraConfig instance
    """
    from peft import LoraConfig, TaskType
    
    target_modules = get_lora_target_modules(model_args.lora_target_modules)
    
    modules_to_save = None
    if model_args.lora_modules_to_save:
        modules_to_save = [module.strip() for module in model_args.lora_modules_to_save.split(",")]
    
    return LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=model_args.use_rslora,
        modules_to_save=modules_to_save,
    )