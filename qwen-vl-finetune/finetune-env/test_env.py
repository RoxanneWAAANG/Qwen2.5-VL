#!/usr/bin/env python3
"""
Qwen2.5-VL 问题诊断脚本
帮助确定具体的问题和解决方案
"""

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