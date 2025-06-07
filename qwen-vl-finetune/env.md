# Qwen2.5-VL -- Linux A100 CUDA 12.7

(qwenvl-new) jack@user:~/Projects/yixin-llm/Qwen2.5-VL/qwen-vl-finetune$ pip show torch flash-attn deepspeed
Name: torch
Version: 2.4.0+cu121
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /home/jack/anaconda3/envs/qwenvl-new/lib/python3.11/site-packages
Requires: filelock, fsspec, jinja2, networkx, nvidia-cublas-cu12, nvidia-cuda-cupti-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, nvidia-cufft-cu12, nvidia-curand-cu12, nvidia-cusolver-cu12, nvidia-cusparse-cu12, nvidia-nccl-cu12, nvidia-nvtx-cu12, sympy, triton, typing-extensions
Required-by: accelerate, bitsandbytes, deepspeed, flash-attn, peft, torchaudio, torchvision
---
Name: flash-attn
Version: 2.6.3
Summary: Flash Attention: Fast and Memory-Efficient Exact Attention
Home-page: https://github.com/Dao-AILab/flash-attention
Author: Tri Dao
Author-email: tri@tridao.me
License: 
Location: /home/jack/anaconda3/envs/qwenvl-new/lib/python3.11/site-packages
Requires: einops, torch
Required-by: 
---
Name: deepspeed
Version: 0.15.4
Summary: DeepSpeed library
Home-page: http://deepspeed.ai
Author: DeepSpeed Team
Author-email: deepspeed-info@microsoft.com
License: Apache Software License 2.0
Location: /home/jack/anaconda3/envs/qwenvl-new/lib/python3.11/site-packages
Requires: hjson, msgpack, ninja, numpy, nvidia-ml-py, packaging, psutil, py-cpuinfo, pydantic, torch, tqdm
Required-by: 

## 步骤 1: 环境清理

```bash
# 完全删除现有conda环境
conda deactivate
conda env remove -n qwenvl-new

# 清理所有缓存
pip cache purge
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/
```

## 步骤 2: 创建新环境

```bash
conda create -n qwenvl-new python=3.11 -y
conda activate qwenvl-new

conda update conda -y
pip install --upgrade pip setuptools wheel
```

## 步骤 3: 配置CUDA环境

```bash
# 检查可用的CUDA版本
ls /usr/local/cuda*/bin/nvcc

# 设置CUDA_HOME (根据你的系统选择12.4或12.6)
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 永久保存到bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证CUDA设置
nvcc --version
nvidia-smi
```

## 步骤 4: 安装PyTorch

```bash
# 安装PyTorch 2.4.0 (支持CUDA 12.1+)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch安装
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU数量: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## 步骤 5: 安装兼容的核心库 (关键步骤)

```bash
# 安装经过测试的兼容版本组合
pip install accelerate==0.34.2
pip install transformers==4.45.2
pip install tokenizers==0.20.0
pip install datasets==3.0.1
pip install peft==0.12.0

# 验证核心库安装
python -c "
import transformers
import accelerate
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')

try:
    from transformers import Trainer
    print('✅ Trainer 导入成功')
except ImportError as e:
    print(f'❌ Trainer 导入失败: {e}')
"
```

## 步骤 6: 安装Flash Attention

```bash
# 安装编译依赖
pip install packaging ninja wheel

# 方法1: 尝试安装Flash Attention (可能需要编译时间)
pip install flash-attn==2.5.9 --no-build-isolation

# 如果编译失败，尝试方法2: 使用预编译版本
# pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases

# 验证Flash Attention (可选，不是必需的)
python -c "
try:
    import flash_attn
    print(f'✅ Flash Attention版本: {flash_attn.__version__}')
except ImportError:
    print('❌ Flash Attention未安装，将使用其他attention实现')
"
```

## 步骤 7: 安装其他依赖

```bash
# 安装深度学习相关库
pip install deepspeed==0.14.4
pip install bitsandbytes==0.43.3

# 安装图像处理库
pip install Pillow==10.4.0
pip install opencv-python==4.10.0.84

# 安装工具库
pip install tqdm
pip install tensorboard
pip install wandb  # 实验追踪 (可选)
pip install scikit-learn
pip install matplotlib seaborn

# 安装Qwen特定依赖
pip install tiktoken
pip install transformers_stream_generator
```

## 步骤 8: 验证Qwen2.5-VL支持

```bash
# 创建完整验证脚本
cat > verify_qwen_setup.py << 'EOF'
#!/usr/bin/env python3
import sys

def check_qwen_support():
    print("=== Qwen2.5-VL 环境验证 ===")
    
    # 基础环境检查
    import torch
    import transformers
    import accelerate
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Transformers版本: {transformers.__version__}")
    print(f"Accelerate版本: {accelerate.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 检查Trainer导入
    try:
        from transformers import Trainer
        print("✅ Trainer 导入成功")
    except ImportError as e:
        print(f"❌ Trainer 导入失败: {e}")
        return False
    
    # 检查Qwen2.5-VL模型支持
    try:
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5VLForConditionalGeneration
        print("✅ Qwen2.5-VL 模型支持正常")
    except ImportError as e:
        print(f"❌ Qwen2.5-VL 模型支持异常: {e}")
        
        # 尝试通用导入方式
        try:
            from transformers import AutoModelForCausalLM
            print("✅ 可以使用AutoModel作为替代")
        except ImportError:
            print("❌ 无法导入任何模型类")
            return False
    
    # 检查Flash Attention
    try:
        import flash_attn
        print(f"✅ Flash Attention可用: {flash_attn.__version__}")
    except ImportError:
        print("⚠️  Flash Attention未安装，将使用标准attention")
    
    # 测试CUDA操作
    if torch.cuda.is_available():
        try:
            x = torch.randn(10, 10).cuda()
            y = torch.randn(10, 10).cuda()
            z = torch.mm(x, y)
            print("✅ CUDA张量操作测试成功")
        except Exception as e:
            print(f"❌ CUDA张量操作测试失败: {e}")
            return False
    
    print("\n🎉 环境配置验证完成!")
    return True

if __name__ == "__main__":
    success = check_qwen_support()
    sys.exit(0 if success else 1)
EOF

# 运行验证
python verify_qwen_setup.py
```

## 步骤 9: 修改训练脚本配置

```bash
# 创建优化的训练配置
cat > training_config.py << 'EOF'
from transformers import TrainingArguments

def get_training_args():
    return TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=2,  # A100适合的batch size
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=3,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        fp16=True,  # 混合精度训练
        dataloader_num_workers=4,
        group_by_length=True,
        report_to="tensorboard",
        run_name="qwen2.5-vl-finetune",
    )

def get_attention_implementation():
    """根据环境返回最佳的attention实现"""
    try:
        import flash_attn
        return "flash_attention_2"
    except ImportError:
        return "sdpa"  # PyTorch的scaled dot product attention
EOF
```

## 步骤 10: 修改你的训练脚本

在你的 `train_qwen.py` 中：

```python
# 导入配置
from training_config import get_training_args, get_attention_implementation

def train():
    # 使用自适应的attention实现
    attn_implementation = get_attention_implementation()
    print(f"使用attention实现: {attn_implementation}")
    
    # 使用优化的训练参数
    training_args = get_training_args()
    
    # 创建trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # ... 其他参数
    )
    
    # 从头开始训练，不使用checkpoint
    trainer.train()  # 确保没有resume_from_checkpoint参数

if __name__ == "__main__":
    train()
```

## 故障排除快速参考

### 如果Transformers导入失败:
```bash
pip uninstall transformers -y
pip install transformers==4.45.2
```

### 如果Flash Attention编译失败:
```bash
# 跳过Flash Attention，使用其他实现
# 在训练脚本中使用: attn_implementation="sdpa"
```

### 如果内存不足:
```bash
# 减小batch size
per_device_train_batch_size=1
gradient_accumulation_steps=16
```

### 如果CUDA版本不匹配:
```bash
# 重新安装匹配的PyTorch版本
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

## 环境保存

```bash
# 保存工作环境
conda env export > qwen-vl-final-environment.yml
pip freeze > requirements-final.txt

# 日后恢复环境
# conda env create -f qwen-vl-final-environment.yml
```

## 最终检查清单

- [ ] Python 3.11 环境创建成功
- [ ] CUDA_HOME 正确设置
- [ ] PyTorch CUDA 功能正常
- [ ] Transformers 和 Accelerate 版本兼容
- [ ] Qwen2.5-VL 模型可以导入
- [ ] 训练脚本去除checkpoint恢复
- [ ] 验证脚本运行成功
