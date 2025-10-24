#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将JSON文件中 "caption_qwen"字段中result: 后的内容合并到captions中
支持命令行传入输入文件和输出文件路径
"""

import json
import os
import argparse

def merge_qwen_captions(input_file, output_file):
    """
    合并Qwen生成的描述到原始captions数组中
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
    """
    # 读取输入文件
    print(f"正在读取输入文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file}'")
        return False
    except json.JSONDecodeError as e:
        print(f"错误: 输入文件不是有效的JSON格式 - {e}")
        return False
    except Exception as e:
        print(f"错误: 读取输入文件时发生错误 - {e}")
        return False
    
    total_items = len(data)
    print(f"总共读取到 {total_items} 个条目")
    
    # 处理每个条目
    processed_count = 0
    for idx, item in enumerate(data):
        if 'caption_qwen' in item and item['caption_qwen'].startswith('result: '):
            # 提取result: 后面的内容
            qwen_caption = item['caption_qwen'][len('result: '):].strip()
            
            # 确保captions字段存在且是列表
            if 'captions' not in item:
                item['captions'] = []
            elif not isinstance(item['captions'], list):
                item['captions'] = [str(item['captions'])]
            
            # 添加Qwen生成的描述到captions数组
            item['captions'].append(qwen_caption)
            processed_count += 1
            
            # 打印进度信息
            if (idx + 1) % 1000 == 0:
                print(f"已处理 {idx + 1}/{total_items} 个条目")
    
    # 保存结果
    try:
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成！结果已保存到: {output_file}")
        print(f"总共处理了 {total_items} 个条目")
        print(f"成功合并了 {processed_count} 个Qwen生成的描述")
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
        description='将JSON文件中caption_qwen字段的result内容合并到captions中',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 默认路径设置为原始的文件路径
    default_input = '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_test.json'
    default_output = '/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_test_merged.json'
    
    parser.add_argument('-i', '--input', '--input_file',
                        type=str,
                        default=default_input,
                        help='输入JSON文件路径')
    
    parser.add_argument('-o', '--output', '--output_file',
                        type=str,
                        default=default_output,
                        help='输出JSON文件路径')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 打印参数信息
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print("="*60)
    
    # 执行合并操作
    success = merge_qwen_captions(args.input, args.output)
    
    if not success:
        print("\n错误: 合并操作失败！")
    else:
        print("\n合并操作成功完成！")


if __name__ == "__main__":
    main()
