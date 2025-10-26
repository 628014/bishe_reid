import os
import json
import random
from pathlib import Path

"""
从realreid每个子目录随机筛选10张图片
"""

# 设置源数据目录和结果保存路径
SOURCE_DIR = '/home/wangrui/code/MLLM4Text-ReID-main/data/RealReid/4.17hhcompany'
OUTPUT_JSON = '/home/wangrui/code/MLLM4Text-ReID-main/result/fliter_10_from_realreid/filtered_images.json'

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg'}

def is_image_file(filename):
    """检查文件是否为图片"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS

def main_filter_images():
    """主函数：从每个子目录随机筛选10张图片"""
    # 创建结果列表
    result = []
    
    # 检查源目录是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"错误：源目录 {SOURCE_DIR} 不存在")
        return
    
    # 获取所有子目录
    subdirectories = [d for d in os.listdir(SOURCE_DIR) 
                     if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    print(f"找到 {len(subdirectories)} 个子目录")
    
    # 遍历每个子目录
    for idx, subdir in enumerate(subdirectories):
        subdir_path = os.path.join(SOURCE_DIR, subdir)
        
        # 获取子目录中的所有图片文件
        image_files = [f for f in os.listdir(subdir_path) 
                      if os.path.isfile(os.path.join(subdir_path, f)) 
                      and is_image_file(f)]
        
        # 随机选择10张图片（如果不足10张则选择全部）
        num_to_select = min(10, len(image_files))
        selected_images = random.sample(image_files, num_to_select)
        
        # 获取绝对路径并添加到结果中
        for img in selected_images:
            img_abs_path = os.path.abspath(os.path.join(subdir_path, img))
            result.append({"image_path": img_abs_path})
        
        # 打印进度
        if (idx + 1) % 50 == 0 or idx + 1 == len(subdirectories):
            print(f"已处理 {idx + 1}/{len(subdirectories)} 个子目录，当前累计 {len(result)} 张图片")
    
    # 保存结果到JSON文件
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n处理完成！")
    print(f"总处理子目录数：{len(subdirectories)}")
    print(f"总选择图片数：{len(result)}")
    print(f"结果已保存到：{OUTPUT_JSON}")


def convert_data_to_llama_factory_format(source_file, output_file):
    """
    将filtered_images.json转换为Llama Factory格式的数据，其中messages的第二个content为空
    """
    # 源文件和目标文件路径
    # source_file = "/home/wangrui/code/MLLM4Text-ReID-main/result/fliter_10_from_realreid/filtered_images.json"
    # output_file = "/home/wangrui/code/LLaMA-Factory/data/mllm_reid_5k_realreid_test.json"
    
    # 确保目标目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # 读取源数据
        print(f"正在读取源文件: {source_file}")
        with open(source_file, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
        
        prompt_template = "<image>\n###Task###\nYou are an expert in the field of Re-Identification (ReID). Please refer to the following requirements to understand the general and specific features of the image, and then combine all the features to return a descriptive language.\n###Requirement###\nChoose one color from black, white, red, purple, yellow, blue, green, pink, gray, and brown.\n1、Gender: male、female\n2、Age: teenager、young、adult、old\n3、Body Build: fat、slightly fat、thin\n4、Length Hair: long hair、medium-length hair、short hair、bald\n5、Wearing hat: yes、no; if yes, the color is: XXX\n6、Carrying backpack: yes、no; if yes, the color is:  XXX\n7、Carrying handbag or bag: yes、no; if yes, the color is:  XXX\n8、Upper Body\n8.1、Sleeve Length: long sleeve、short sleeve\n8.2、Inner Lining: yes、no; if yes, the color is:\n8.3、Color of upper-body: XXX\n9、Lower Body\n9.1、Length of lower-body: long lower-body clothing、short\n9.2、Type of lower-body: dress、pants\n9.3、Color of lower-body: XXX\n10、Shoe Color: XXX\n11、Emotion: Happy、Surprised、Sad、Angry、Disgusted、Fearful、Neutral、Other\n12、Gait and Posture: XXX\n###Output###\nCombine all the attributes above into a natural language as the final output.\n",
        
        # 转换数据格式
        print(f"正在转换数据格式，共处理 {len(source_data)} 个图像...")
        llama_factory_data = []
        
        for i, item in enumerate(source_data):
            # 进度显示
            if (i + 1) % 1000 == 0 or i + 1 == len(source_data):
                print(f"已处理 {i + 1}/{len(source_data)} 个图像")
            
            # 创建Llama Factory格式的数据项
            llama_item = {
                "messages": [
                    {
                        "content": prompt_template,
                        "role": "user"
                    },
                    {
                        "content": "",  
                        "role": "assistant"
                    }
                ],
                "images": [item["image_path"]]  # 图像路径数组
            }
            
            llama_factory_data.append(llama_item)
        
        # 保存转换后的数据
        print(f"正在保存结果到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(llama_factory_data, f, ensure_ascii=False, indent=2)
        
        print(f"数据转换完成！共处理 {len(llama_factory_data)} 个图像")
        print(f"结果文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise
if __name__ == "__main__":
    # 设置随机种子以确保结果可复现（可选）
    # random.seed(42)
    # 随机筛选10张
    # main_filter_images()
    # 筛选后输出成Llama Factory格式
    output_file = "/home/wangrui/code/LLaMA-Factory/data/mllm_reid_5k_realreid_test.json"
    convert_data_to_llama_factory_format(OUTPUT_JSON, output_file)
