import json
import jsonlines
import os

def convert_jsonl_to_captions(jsonl_path, output_json_path):
    """
    将generated_predictions_with_images.jsonl转换为data_captions.json格式
    
    Args:
        jsonl_path (str): 输入JSONL文件路径
        output_json_path (str): 输出JSON文件路径
    """
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    captions_data = []
    
    with jsonlines.open(jsonl_path, 'r') as reader:
        for line_num, item in enumerate(reader, 1):
            try:
                # 提取必要字段
                images_path = item.get('images_path')
                predict = item.get('predict')
                
                if not images_path or not predict:
                    print(f"警告: 第{line_num}行缺少images_path或predict字段，已跳过")
                    continue
                
                # 处理images_path可能是列表的情况
                if isinstance(images_path, list):
                    # 如果是列表，尝试获取第一个元素
                    if images_path and isinstance(images_path[0], str):
                        images_path = images_path[0]
                    else:
                        print(f"警告: 第{line_num}行images_path是列表但格式无效，已跳过")
                        continue
                
                # 确保images_path是字符串
                if not isinstance(images_path, str):
                    print(f"警告: 第{line_num}行images_path不是字符串类型，已跳过")
                    continue
                
                # 从文件名中提取id（第一个下划线前的部分）
                filename = os.path.basename(images_path)
                id_part = filename.split('_')[0]
                if not id_part.isdigit():
                    print(f"警告: 第{line_num}行文件名{filename}无法提取有效id，已跳过")
                    continue
                
                # 构建输出条目
                caption_entry = {
                    "id": int(id_part),
                    "img_path": images_path,
                    "captions": [predict],  # 确保captions是列表格式
                    "split": "test"
                }
                
                captions_data.append(caption_entry)
                
            except Exception as e:
                print(f"处理第{line_num}行时出错: {str(e)}，已跳过")
                continue
    
    # 写入输出JSON文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(captions_data, f, indent=2, ensure_ascii=False)
    
    print(f"转换完成，共处理{len(captions_data)}条记录，输出文件: {output_json_path}")

if __name__ == "__main__":
    # 配置文件路径
    input_jsonl = "/home/wangrui/code/LLaMA-Factory/train_reid/data/local_data/infer_res/lora64_drop005_realreid_5k/generated_predictions_with_images.jsonl"
    output_json = "/home/wangrui/code/MLLM4Text-ReID-main/data/RealReid/random_10_all_5k/data_captions.json"
    
    # 执行转换
    convert_jsonl_to_captions(input_jsonl, output_json)