"""
1、处理data/RSTPReid/data_captions.json文件，按照"split": "train"进行筛选，将筛选结果保存到data/RSTPReid/data_captions_train.json文件中
2、将data_captions_train.json文件进行解析，其中"img_path"解析出来，拼接上“/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/”，当作get_qwen_call_local中的local_path
3、调用get_qwen_call_local函数，将local_path作为参数传入，获取到qwen的回复，同时将data_captions_train.json中的captions字段拼接作为prompt中的caption字段
4、将qwen的回复保存到/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_train.json文件中，对应的"caption"字段
5、注意并发操作，高效调用get_qwen_call_local，实时查看处理到data_captions_train.json中的第几条了

"""

import json
import os
import concurrent.futures
import threading
from tqdm import tqdm
from qwen_api import get_qwen_call_local
from prompt import prompt

DATA_CAPTIONS_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions.json"
DATA_CAPTIONS_TRAIN_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions_train.json"
OUTPUT_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/data_captions_train.json"
IMAGES_BASE_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/"


os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def filter_train_data():

    print(f"正在读取并筛选数据: {DATA_CAPTIONS_PATH}")
    with open(DATA_CAPTIONS_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = [item for item in data if item.get('split') == 'train']
    print(f"筛选出的train数据数量: {len(train_data)}")
    
    # 保存筛选结果
    with open(DATA_CAPTIONS_TRAIN_PATH, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"train数据已保存到: {DATA_CAPTIONS_TRAIN_PATH}")
    
    return train_data

def get_train_data():
    """获取train数据"""
    with open(DATA_CAPTIONS_TRAIN_PATH, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    return train_data

def process_item(item, index, total):
    """处理单个数据项"""
    try:
        
        img_path = item.get('img_path', '')
        if not img_path:
            return index, None
        
        local_path = os.path.join(IMAGES_BASE_PATH, img_path)
        
        # 检查图片文件是否存在
        if not os.path.exists(local_path):
            print(f"警告: 图片文件不存在: {local_path}")
            return index, None
        
        # 构建自定义prompt，将caption替换到占位符
        caption = item.get('captions', '')
        caption_all = ' '.join(caption)
        custom_prompt = prompt.format(caption=caption_all)
        # print(f"第{index+1}/{total}项 - 自定义prompt: {custom_prompt}")
        # 调用API获取回复
        qwen_response = get_qwen_call_local(local_path, custom_prompt)
        
        # 更新item
        if qwen_response:
            item['caption_qwen'] = qwen_response
            return index, item
        else:
            return index, None
            
    except Exception as e:
        print(f"处理第{index+1}/{total}项时出错: {e}")
        return index, None

def load_existing_results():
    """加载已存在的结果并返回已处理项的img_path集合"""
    processed_img_paths = set()
    existing_results = []
    
    if os.path.exists(OUTPUT_PATH):
        print(f"检测到已存在的结果文件，正在加载: {OUTPUT_PATH}")
        try:
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # 提取已处理的img_path
            for item in existing_results:
                if 'img_path' in item:
                    processed_img_paths.add(item['img_path'])
            
            print(f"已加载{len(existing_results)}条已处理结果，{len(processed_img_paths)}个唯一图片路径")
        except Exception as e:
            print(f"加载现有结果时出错: {e}")
    
    return processed_img_paths, existing_results

def save_item(item, file_lock):
    """安全地保存单个处理结果到输出文件"""
    with file_lock:
        # 读取现有内容
        existing_data = []
        if os.path.exists(OUTPUT_PATH):
            try:
                with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except Exception as e:
                print(f"读取现有数据时出错: {e}")
                existing_data = []
        
        # 添加新项
        existing_data.append(item)
        
        # 保存更新后的数据
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

def main():
   
    processed_img_paths, existing_results = load_existing_results()
    
    # train_data = filter_train_data()
    train_data = get_train_data()
    
    # 前100条实验
    train_data = train_data[:5000]
    
    # 过滤掉已处理的项
    filtered_train_data = []
    for item in train_data:
        if 'img_path' in item and item['img_path'] not in processed_img_paths:
            filtered_train_data.append(item)
    
    total_processed = len(existing_results)
    total_to_process = len(filtered_train_data)
    
    print(f"已处理{total_processed}项，还需处理{total_to_process}项")
    
    if total_to_process == 0:
        print("所有数据都已处理完成，无需继续")
        return
    
    # 创建文件锁，确保并发写入安全
    file_lock = threading.Lock()
    
    print(f"开始并发处理{total_to_process}项数据...")
    max_workers = 10 # 控制并发数，避免API限制
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {executor.submit(process_item, item, idx, total_to_process): idx 
                         for idx, item in enumerate(filtered_train_data)}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=total_to_process):
            index, result = future.result()
            if result:
                # 立即保存处理结果
                save_item(result, file_lock)
                total_processed += 1
                print(f"已处理并保存: {total_processed}项")
    
    print(f"所有数据处理完成！总共处理: {total_processed}项")
    print(f"最终结果保存在: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()