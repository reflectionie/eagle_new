#!/usr/bin/env python3
import os
import re

def get_folder_start_num(dirname):
    """
    从文件夹名中提取开头编号，比如 "sharegpt_0_67999_mufp16" 提取 0
    """
    m = re.match(r'sharegpt_(\d+)_67999_mufp16', dirname)
    if m:
        return int(m.group(1))
    else:
        return None

def main():
    # 获取当前工作目录
    base_dir = os.getcwd()

    # 找出当前目录下所有符合 "sharegpt_*_67999_mufp16" 格式的文件夹
    folders = [d for d in os.listdir(base_dir) 
               if os.path.isdir(d) and d.startswith("sharegpt_") and d.endswith("_67999_mufp16")]

    # 根据文件夹名称中的数字排序（例如：0, 50057, 67626）
    folders.sort(key=lambda d: get_folder_start_num(d))

    # 全局编号计数器，初始为 0
    global_counter = 0

    # 正则表达式用于匹配 ckpt 文件名，比如 data_123.ckpt
    file_pattern = re.compile(r'data_(\d+)\.ckpt')

    for folder in folders:
        # 构造子文件夹 "0" 的路径
        subfolder = os.path.join(base_dir, folder, "0")
        if not os.path.exists(subfolder):
            print(f"子文件夹 {subfolder} 不存在，跳过。")
            continue

        print(f"正在处理文件夹： {folder}/0")
        # 列出该子文件夹中的所有文件
        files = os.listdir(subfolder)

        # 将匹配的文件按原有序号提取出来，并按数字顺序排序
        file_entries = []
        for fname in files:
            m = file_pattern.fullmatch(fname)
            if m:
                num = int(m.group(1))
                file_entries.append((num, fname))
        file_entries.sort(key=lambda x: x[0])

        # 遍历所有文件，按全局计数器赋予新的编号
        for old_num, fname in file_entries:
            old_path = os.path.join(subfolder, fname)
            new_fname = f"data_{global_counter}.ckpt"
            new_path = os.path.join(subfolder, new_fname)
            # 如果原文件名与新文件名不一样，则重命名
            if old_path != new_path:
                print(f"重命名: {old_path} -> {new_path}")
                os.rename(old_path, new_path)
            else:
                print(f"文件名 {old_path} 已经正确，无需更改。")
            global_counter += 1

    print("所有文件重命名完成。")

if __name__ == "__main__":
    main()

