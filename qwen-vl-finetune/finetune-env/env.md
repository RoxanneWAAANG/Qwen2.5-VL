# Qwen2.5-VL -- Linux A100 CUDA 12.7

## Step 1 -- Create New Environment

```bash
conda deactivate
conda env remove -n qwenvl

# clean cache.
pip cache purge
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/
```

```bash
conda create -n qwenvl-new python=3.11 -y
conda activate qwenvl-new

conda update conda -y
pip install --upgrade pip setuptools wheel
```

## Step 2 -- Setup CUDA Environment

```bash
# 检查可用的CUDA版本
ls /usr/local/cuda*/bin/nvcc

# 设置CUDA_HOME (根据你的系统选择12.4或12.6)
# replace the path after export CUDA_HOME=
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

## Step 3 -- Install PyTorch

```bash
# PyTorch 2.4.0 (CUDA 12.1+)
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 验证PyTorch安装
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"
```

## Step 4 -- Install Transformers

Here I refer to issue from official repo: https://github.com/QwenLM/Qwen2.5-VL/issues/936.

```bash
pip install accelerate==1.8.0.dev0
pip install transformers==4.49.0
pip install tokenizers==0.21.1
pip install datasets==3.2.0
pip install peft==0.13.2

python -c "
import transformers
import accelerate
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')

try:
    from transformers import Trainer
    print('Trainer Import Success!')
except ImportError as e:
    print(f'Trainer Import Failed: {e}')
"
```

## Step 5 -- Install Flash-Attention

```bash
pip install packaging ninja wheel

# 方法1: 尝试安装Flash Attention (可能需要15min编译时间)
pip install flash-attn==2.6.3 --no-build-isolation

# 如果编译失败，尝试方法2: 使用预编译版本
# pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases

python -c "
try:
    import flash_attn
    print(f'Flash Attention Version: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention Not Installed!')
"
```

```bash
# My Install Log: 
(qwenvl-new) $ pip show torch flash-attn deepspeed
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
```

## Step 6 -- Install Other Dependencies

```bash
pip install deepspeed==0.15.4
pip install bitsandbytes==0.44.1

# 安装图像处理库
pip install Pillow==10.4.0
pip install opencv-python==4.10.0.84

# 安装工具库
pip install tqdm==4.67.1
pip install tensorboard==2.19.0
pip install scikit-learn
pip install matplotlib seaborn
pip install wandb # 我在这里没用，optional

# 安装Qwen特定依赖
pip install tiktoken==0.9.0
pip install transformers_stream_generator==0.0.5
```

## Step 7 -- Verification

``bash
python3 qwen-vl-finetune/finetune-env/test_env.py
```

```bash
# here is Chinese version.
cat > verify_qwen_setup.py << 'EOF'
#!/usr/bin/env python3
#!/usr/bin/env python3
import sys
import importlib
import traceback

def check_basic_environment():
    """检查基础环境"""
    print("=== 基础环境检查 ===")
    
    import torch
    import transformers
    
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"Transformers版本: {transformers.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")

def check_transformers_modules():
    """检查 transformers 模块结构"""
    print("\n=== Transformers 模块结构检查 ===")
    
    import transformers
    
    # 检查是否有 models 目录
    try:
        import transformers.models
        print("✅ transformers.models 存在")
        
        # 列出可用的模型
        models_dir = transformers.models.__path__[0]
        import os
        available_models = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        
        print(f"可用模型数量: {len(available_models)}")
        
        # 检查是否有 qwen 相关模型
        qwen_models = [m for m in available_models if 'qwen' in m.lower()]
        print(f"Qwen相关模型: {qwen_models}")
        
    except Exception as e:
        print(f"❌ transformers.models 检查失败: {e}")

def check_qwen_models():
    """检查 Qwen 模型支持"""
    print("\n=== Qwen 模型支持检查 ===")
    
    # 尝试不同的导入方式
    import_attempts = [
        ("直接导入 qwen2_5_vl", "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl", "Qwen2_5VLForConditionalGeneration"),
        ("导入 qwen2_vl", "transformers.models.qwen2_vl.modeling_qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("AutoModel 导入", "transformers", "AutoModelForCausalLM"),
        ("AutoModel VL 导入", "transformers", "AutoModelForVision2Seq"),
    ]
    
    successful_imports = []
    
    for name, module_path, class_name in import_attempts:
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            print(f"✅ {name}: {class_name} 可用")
            successful_imports.append((name, module_path, class_name))
        except ImportError as e:
            print(f"❌ {name}: {e}")
        except AttributeError as e:
            print(f"❌ {name}: 模块存在但类不存在 - {e}")
        except Exception as e:
            print(f"❌ {name}: 未知错误 - {e}")
    
    return successful_imports

def check_model_loading(successful_imports):
    """测试模型加载"""
    print("\n=== 模型加载测试 ===")
    
    if not successful_imports:
        print("❌ 没有可用的模型类，跳过加载测试")
        return
    
    # 使用第一个成功的导入进行测试
    name, module_path, class_name = successful_imports[0]
    print(f"使用 {name} 进行测试...")
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        
        # 测试 from_pretrained 是否可用
        if hasattr(model_class, 'from_pretrained'):
            print("✅ from_pretrained 方法可用")
        else:
            print("❌ from_pretrained 方法不可用")
            
    except Exception as e:
        print(f"❌ 模型类测试失败: {e}")

def check_attention_implementations():
    """检查 attention 实现"""
    print("\n=== Attention 实现检查 ===")
    
    # 检查 Flash Attention
    try:
        import flash_attn
        print(f"✅ Flash Attention: {flash_attn.__version__}")
        
        # 检查核心函数
        try:
            from flash_attn import flash_attn_func
            print("✅ flash_attn_func 可用")
        except ImportError:
            print("❌ flash_attn_func 不可用")
            
    except ImportError:
        print("❌ Flash Attention 未安装")
    
    # 检查 PyTorch SDPA
    try:
        import torch.nn.functional as F
        if hasattr(F, 'scaled_dot_product_attention'):
            print("✅ PyTorch SDPA 可用")
        else:
            print("❌ PyTorch SDPA 不可用")
    except Exception as e:
        print(f"❌ SDPA 检查失败: {e}")

def suggest_solutions(successful_imports):
    """建议解决方案"""
    print("\n=== 建议解决方案 ===")
    
    if not successful_imports:
        print("🔧 主要问题：Qwen2.5-VL 模型类不可用")
        print("\n推荐解决方案：")
        print("1. 升级到支持 Qwen2.5-VL 的 transformers 版本：")
        print("   pip install transformers>=4.46.0")
        print("2. 或者使用开发版本：")
        print("   pip install git+https://github.com/huggingface/transformers.git")
        print("3. 检查是否使用了正确的模型名称")
        
    else:
        print("✅ 发现可用的模型类:")
        for name, module_path, class_name in successful_imports:
            print(f"   - {name}: {module_path}.{class_name}")
        
        print("\n推荐使用策略：")
        print("1. 优先使用 Qwen2_5VLForConditionalGeneration")
        print("2. 如果不可用，使用 AutoModelForCausalLM")
        print("3. 设置 trust_remote_code=True")

def generate_working_import_code(successful_imports):
    """生成可工作的导入代码"""
    print("\n=== 生成可工作的导入代码 ===")
    
    if successful_imports:
        name, module_path, class_name = successful_imports[0]
        
        print("在你的代码中使用以下导入：")
        print("```python")
        print(f"# 方法1: 直接导入")
        print(f"from {module_path} import {class_name}")
        print()
        print("# 方法2: 安全导入")
        print("try:")
        print(f"    from {module_path} import {class_name}")
        print("except ImportError:")
        print("    from transformers import AutoModelForCausalLM as " + class_name)
        print("```")
    else:
        print("建议使用通用导入：")
        print("```python")
        print("from transformers import AutoModelForCausalLM")
        print("from transformers import AutoTokenizer, AutoProcessor")
        print()
        print("# 加载模型时使用")
        print("model = AutoModelForCausalLM.from_pretrained(")
        print("    model_path,")
        print("    trust_remote_code=True,")
        print("    torch_dtype=torch.float16")
        print(")")
        print("```")

def main():
    """主诊断流程"""
    print("🔍 开始 Qwen2.5-VL 环境诊断...\n")
    
    try:
        # 基础环境检查
        check_basic_environment()
        
        # 模块结构检查
        check_transformers_modules()
        
        # Qwen 模型检查
        successful_imports = check_qwen_models()
        
        # 模型加载测试
        check_model_loading(successful_imports)
        
        # Attention 实现检查
        check_attention_implementations()
        
        # 建议解决方案
        suggest_solutions(successful_imports)
        
        # 生成工作代码
        generate_working_import_code(successful_imports)
        
    except Exception as e:
        print(f"\n❌ 诊断过程中出错: {e}")
        print("详细错误信息:")
        traceback.print_exc()
    
    print("\n🎯 诊断完成！")

if __name__ == "__main__":
    main()
    sys.exit(0 if success else 1)
EOF

# 运行验证
python verify_qwen_setup.py
```


## Appendix -- Save Environment

```bash
conda env export > qwenvl-env.yml
pip freeze > requirements.txt

# 恢复环境
# conda env create -f qwen-vl-final-environment.yml
```

