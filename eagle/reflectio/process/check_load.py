import os
import glob
import torch
import concurrent.futures

def load_file(file_path):
    """
    尝试加载给定路径的 ckpt 文件，如果加载失败，则返回错误信息。
    """
    try:
        # 加载时使用 CPU 设备，防止因 GPU 内存不足出错
        _ = torch.load(file_path, map_location=torch.device("cpu"))
        return file_path, None
    except Exception as e:
        return file_path, e

def main():
    # 指定包含 ckpt 文件的目录
    data_dir = "/home/5/uu02155/data/llama/eagle_new/eagle/reflectio/train_data/sharegpt_0_67999_mufp16/0"
    # 构造文件查找模式（假设文件名为 data_数字编号.ckpt）
    pattern = os.path.join(data_dir, "data_*.ckpt")
    files = glob.glob(pattern)
    
    print(f"共找到 {len(files)} 个 ckpt 文件。")

    # 根据你的描述：4 张卡，每张卡最多 6 个 batch，
    # 因此我们并行最多开启 4*6=24 个进程来加载文件。
    max_workers = 24

    failed_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，并实时获取结果
        futures = {executor.submit(load_file, f): f for f in files}
        for future in concurrent.futures.as_completed(futures):
            file_path, err = future.result()
            if err is not None:
                print(f"加载失败：{file_path}，错误信息：{err}")
                failed_files.append(file_path)
            else:
                print(f"加载成功：{file_path}")

    print("\n加载失败的文件列表：")
    for f in failed_files:
        print(f)

if __name__ == "__main__":
    main()
