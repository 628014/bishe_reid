#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并多个JSON文件为一个文件
输入文件为JSON数组格式，将所有数组元素合并到一个新的JSON数组中
"""

import json
import os
import argparse
import sys

def merge_json_files(input_files, output_file):
    """
    合并多个JSON文件为一个文件
    
    Args:
        input_files: 输入JSON文件路径列表
        output_file: 输出JSON文件路径
        
    Returns:
        bool: 合并是否成功
    """
    print(f"开始合并JSON文件...")
    print(f"输入文件列表: {', '.join(input_files)}")
    print(f"输出文件: {output_file}")
    print("=" * 80)
    
    all_data = []
    total_items = 0
    
    # 处理每个输入文件
    for idx, input_file in enumerate(input_files):
        print(f"正在处理文件 {idx + 1}/{len(input_files)}: {input_file}")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(input_file):
                print(f"错误: 文件不存在 - {input_file}")
                return False
            
            # 读取JSON文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据是否为列表格式
            if not isinstance(data, list):
                print(f"错误: 文件 {input_file} 不是有效的JSON数组格式")
                return False
            
            # 添加到总数据中
            all_data.extend(data)
            items_count = len(data)
            total_items += items_count
            print(f"  成功读取 {items_count} 个条目")
            
        except json.JSONDecodeError as e:
            print(f"错误: 文件 {input_file} 格式错误 - {e}")
            return False
        except Exception as e:
            print(f"错误: 处理文件 {input_file} 时发生错误 - {e}")
            return False
    
    print("=" * 80)
    print(f"所有文件处理完成！")
    print(f"总共合并了 {total_items} 个条目")
    
    # 删除所有条目中的caption_qwen键
    caption_qwen_count = 0
    for item in all_data:
        if 'caption_qwen' in item:
            del item['caption_qwen']
            caption_qwen_count += 1
    print(f"已删除 {caption_qwen_count} 个条目中的caption_qwen键")
    
    # 保存合并后的结果
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 写入合并后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n合并成功！")
        print(f"结果已保存到: {output_file}")
        print(f"输出文件包含 {len(all_data)} 个条目")
        return True
        
    except Exception as e:
        print(f"错误: 保存输出文件时发生错误 - {e}")
        return False

def parse_args():
    """
    解析命令行参数
    
    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description='合并多个JSON数组文件为一个文件',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 设置默认的输入和输出文件
    default_input_files = [
        '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_test_merged.json',
        '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_train_merged.json',
        '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_val_merged.json'
    ]
    default_output_file = '/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions.json'
    
    parser.add_argument('-i', '--input', '--input_files',
                        type=str, 
                        nargs='+',
                        default=default_input_files,
                        help='输入JSON文件路径列表')
    
    parser.add_argument('-o', '--output', '--output_file',
                        type=str,
                        default=default_output_file,
                        help='输出JSON文件路径')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 执行合并操作
    success = merge_json_files(args.input, args.output)
    
    if not success:
        print("\n错误: JSON文件合并失败！")
        sys.exit(1)
    else:
        print("\nJSON文件合并成功完成！")
        sys.exit(0)


if __name__ == "__main__":
    main()