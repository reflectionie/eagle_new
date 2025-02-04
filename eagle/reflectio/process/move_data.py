#!/usr/bin/env python3
import os
import shutil

def main():
    # 当前目录
    base_dir = os.getcwd()
    
    # 定义源文件夹列表（需要移动的文件夹）
    source_folders = [
        "sharegpt_50057_67999_mufp16",
        "sharegpt_67626_67999_mufp16"
    ]
    
    # 定义目标文件夹（文件将被移动到该文件夹下的 "0" 子目录中）
    target_folder = "sharegpt_0_67999_mufp16"
    
    # 构造目标子文件夹 "0" 的路径
    target_subfolder = os.path.join(base_dir, target_folder, "0")
    if not os.path.exists(target_subfolder):
        print(f"目标子文件夹 {target_subfolder} 不存在，请检查目录结构。")
        return

    # 遍历每个源文件夹
    for folder in source_folders:
        source_subfolder = os.path.join(base_dir, folder, "0")
        if not os.path.exists(source_subfolder):
            print(f"源子文件夹 {source_subfolder} 不存在，跳过。")
            continue
        
        print(f"正在处理源文件夹: {source_subfolder}")
        # 列出该文件夹内所有文件
        for fname in os.listdir(source_subfolder):
            # 这里简单判断文件名是否以 "data_" 开头，并以 ".ckpt" 结尾
            if fname.startswith("data_") and fname.endswith(".ckpt"):
                src_path = os.path.join(source_subfolder, fname)
                dst_path = os.path.join(target_subfolder, fname)
                print(f"移动文件: {src_path} -> {dst_path}")
                # 执行移动操作
                shutil.move(src_path, dst_path)
    
    print("文件移动完成。")

if __name__ == "__main__":
    main()

