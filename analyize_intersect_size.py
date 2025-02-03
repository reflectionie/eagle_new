#!/usr/bin/env python3
import json
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_jsonl_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        sys.exit(1)

    total_sum = 0
    total_count = 0
    
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                file_sum, file_count = process_data(data)
                total_sum += file_sum
                total_count += file_count
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
    
    if total_count > 0:
        overall_avg = total_sum / total_count
    else:
        overall_avg = 0
    
    print(f"Overall average intersect: {overall_avg:.2f}")

def process_data(data):
    # 预期数据格式: { "seq_0": [int, int, ...], "seq_1": [...], ... }
    file_sum = 0
    file_count = 0
    
    for seq_id, intersect_list in data.items():
        seq_length = len(intersect_list)
        avg_intersect = sum(intersect_list) / seq_length if seq_length > 0 else 0
        print(f"{seq_id}: length = {seq_length}, average intersect = {avg_intersect:.2f}")
        
        file_sum += sum(intersect_list)
        file_count += seq_length
    
    return file_sum, file_count

if __name__ == "__main__":
    main()