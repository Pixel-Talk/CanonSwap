import torch
import os



def extract_transfer_refine_from_tar(tar_path):
    """
    从pth.tar文件中提取transfer和refine字段并保存为pth文件
    """
    if not os.path.exists(tar_path):
        print(f"错误: 文件 {tar_path} 不存在")
        return

    print(f"正在从 {tar_path} 提取transfer和refine字段...")

    # 加载tar文件
    checkpoint = torch.load(tar_path, map_location=torch.device("cpu"))

    # 获取文件目录
    base_dir = os.path.dirname(tar_path)
    base_name = os.path.splitext(os.path.basename(tar_path))[0]

    # 提取并保存transfer权重
    if 'transfer' in checkpoint:
        transfer_weights = checkpoint['transfer']
        transfer_path = os.path.join(base_dir, f"{base_name}_transfer.pth")
        torch.save(transfer_weights, transfer_path)
        print(f"transfer权重已保存到: {transfer_path}")
        print(f"transfer权重包含 {len(transfer_weights)} 个参数")
    else:
        print("警告: 在checkpoint中未找到'transfer'字段")

    # 提取并保存refine权重
    if 'refine' in checkpoint:
        refine_weights = checkpoint['refine']
        refine_path = os.path.join(base_dir, f"{base_name}_refine.pth")
        torch.save(refine_weights, refine_path)
        print(f"refine权重已保存到: {refine_path}")
        print(f"refine权重包含 {len(refine_weights)} 个参数")
    else:
        print("警告: 在checkpoint中未找到'refine'字段")

    # 显示所有可用的键
    print(f"原文件中的所有键: {list(checkpoint.keys())}")

def create_combined_weights(tar_path):
    """
    将所有权重合并保存到一个pth文件中
    """
    if not os.path.exists(tar_path):
        print(f"错误: 文件 {tar_path} 不存在")
        return

    print("正在创建合并权重文件...")

    # 定义所有权重文件路径
    base_models_dir = r"D:\Projects\canonswap\Code\CanonSwap2\pretrained_weights\liveportrait\base_models"

    weight_files = {
        'appearance_feature_extractor': os.path.join(base_models_dir, 'appearance_feature_extractor.pth'),
        'motion_extractor': os.path.join(base_models_dir, 'student_motion_distill_e005.pth'),
        'warping_module': os.path.join(base_models_dir, 'warping_module.pth'),
        'spade_generator': os.path.join(base_models_dir, 'spade_generator.pth')
    }

    # 创建合并权重字典
    combined_weights = {}

    # 加载基础模型权重
    for model_name, file_path in weight_files.items():
        if os.path.exists(file_path):
            print(f"正在加载 {model_name} 从 {file_path}")
            if model_name == 'motion_extractor':
                # motion_extractor需要提取model字段
                checkpoint = torch.load(file_path, map_location=torch.device("cpu"))
                if 'model' in checkpoint:
                    combined_weights[model_name] = checkpoint['model']
                    print(f"  - 提取了motion_extractor的model字段，包含 {len(checkpoint['model'])} 个参数")
                else:
                    print(f"  - 警告: 在{file_path}中未找到'model'字段")
            else:
                # 其他模型直接加载
                weights = torch.load(file_path, map_location=torch.device("cpu"))
                combined_weights[model_name] = weights
                print(f"  - 加载了{model_name}，包含 {len(weights)} 个参数")
        else:
            print(f"警告: 文件 {file_path} 不存在")

    # 加载tar文件中的transfer和refine权重
    print(f"正在从 {tar_path} 加载transfer和refine权重...")
    checkpoint = torch.load(tar_path, map_location=torch.device("cpu"))

    if 'transfer' in checkpoint:
        combined_weights['transfer'] = checkpoint['transfer']
        print(f"  - 加载了transfer权重，包含 {len(checkpoint['transfer'])} 个参数")
    else:
        print("警告: 在checkpoint中未找到'transfer'字段")

    if 'refine' in checkpoint:
        combined_weights['refine'] = checkpoint['refine']
        print(f"  - 加载了refine权重，包含 {len(checkpoint['refine'])} 个参数")
    else:
        print("警告: 在checkpoint中未找到'refine'字段")

    # 保存合并权重
    output_path = r"D:\Projects\canonswap\Code\CanonSwap2\pretrained_weights\combined_weights.pth"
    torch.save(combined_weights, output_path)

    print(f"\n合并权重已保存到: {output_path}")
    print(f"合并权重包含以下模块: {list(combined_weights.keys())}")

    # 显示每个模块的参数数量
    total_params = 0
    for model_name, weights in combined_weights.items():
        param_count = len(weights)
        total_params += param_count
        print(f"  - {model_name}: {param_count} 个参数")

    print(f"总参数数量: {total_params}")
    return output_path

def main():
    print("=== 合并权重脚本 ===")
    print()

    # 获取tar文件路径
    tar_file_path = input("请输入包含transfer和refine权重的pth.tar文件完整路径: ").strip()

    if tar_file_path and os.path.exists(tar_file_path):
        create_combined_weights(tar_file_path)
    else:
        print("错误: 请提供有效的tar文件路径")
        print("你也可以直接调用: create_combined_weights('你的文件路径.pth.tar')")

    print("\n脚本执行完成！")

if __name__ == "__main__":
    main()
