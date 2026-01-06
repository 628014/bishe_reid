import json
import os
import concurrent.futures
import threading
from tqdm import tqdm
from qwen_api import get_qwen_call_local
from prompt_sim import prompt

# 配置文件路径
DATA_CAPTIONS_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/data_captions.json"
OUTPUT_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/process/data_res/get_sim_data/data_caption_all_qwen.json"
IMAGES_BASE_PATH = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/"

# 创建输出目录
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def load_data(path):
    """加载数据"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, path):
    """保存数据"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_results(path):
    """加载已存在的结果并返回已处理项的img_path集合"""
    processed_img_paths = set()
    existing_results = []
    
    if os.path.exists(path):
        print(f"检测到已存在的结果文件，正在加载: {path}")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            
            # 提取已处理的img_path
            for item in existing_results:
                if 'img_path' in item:
                    processed_img_paths.add(item['img_path'])
            
            print(f"已加载{len(existing_results)}条已处理结果，{len(processed_img_paths)}个唯一图片路径")
        except Exception as e:
            print(f"加载现有结果时出错: {e}")
    
    return processed_img_paths, existing_results


def parse_qwen_response(response):
    """解析通义千问的回复，提取匹配分数和原因
    Args:
        response: 模型返回的文本内容
    
    Returns:
        tuple: (提取的匹配分数列表, 提取的原因列表)
    """
    try:
        scores = []
        reasons = []
        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('1、') or line.startswith('2、') or line.startswith('3、'):
                # 提取匹配分数和原因
                if 'matching score:' in line and 'Reason:' in line:
                    # 分割分数和原因
                    parts = line.split(';')
                    if len(parts) >= 2:
                        # 提取分数
                        score_part = parts[0].split('matching score:')[1].strip()
                        score = ''.join(filter(lambda x: x.isdigit() or x == '.', score_part))
                        scores.append(score)
                        # 提取原因
                        reason_part = ';'.join(parts[1:]).split('Reason:')[1].strip()
                        reasons.append(reason_part)
        return scores, reasons
    except Exception as e:
        print(f"解析模型回复时出错: {e}")
        return [], []


def process_item(item, index, total):
    """处理单个数据项
    Args:
        item: 数据项
        index: 索引
        total: 总数
    
    Returns:
        tuple: (index, 更新后的item或None)
    """
    try:
        img_path = item.get('img_path', '')
        if not img_path:
            return index, None
        
        local_path = os.path.join(IMAGES_BASE_PATH, img_path)
        
        # 检查图片文件是否存在
        if not os.path.exists(local_path):
            print(f"警告: 图片文件不存在: {local_path}")
            return index, None
        
        # 检查是否已经处理过
        if 'match_score' in item and item['match_score']:
            print(f"第{index+1}/{total}项 - 已处理过，跳过")
            return index, item
        
        # 构建自定义prompt，将三个caption替换到占位符
        captions = item.get('captions', [])
        if len(captions) < 3:
            print(f"警告: 第{index+1}/{total}项 - 描述数量不足3个: {len(captions)}")
            return index, None
        
        custom_prompt = prompt.format(
            caption1=captions[0],
            caption2=captions[1],
            caption3=captions[2]
        )
        
        # 调用API获取回复
        qwen_response = get_qwen_call_local(local_path, custom_prompt)
        
        # 更新item
        if qwen_response:
            # 解析回复，提取匹配分数和原因
            match_scores, reasons = parse_qwen_response(qwen_response)
            if len(match_scores) == 3 and len(reasons) == 3:
                item['match_score'] = match_scores
                item['reason'] = reasons
                return index, item
            else:
                print(f"警告: 第{index+1}/{total}项 - 解析匹配分数或原因失败，得到{len(match_scores)}个分数和{len(reasons)}个原因")
                return index, None
        else:
            return index, None
            
    except Exception as e:
        print(f"处理第{index+1}/{total}项时出错: {e}")
        return index, None


def save_item(item, file_lock, path):
    """安全地保存单个处理结果到输出文件"""
    with file_lock:
        # 读取现有内容
        existing_results = []
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except Exception as e:
                print(f"读取现有数据时出错: {e}")
                existing_results = []
        
        # 添加新项
        existing_results.append(item)
        
        # 保存更新后的数据
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=2)


def main():
    print(f"开始处理数据: {DATA_CAPTIONS_PATH}")
    
    # 加载已存在的结果
    processed_img_paths, existing_results = load_existing_results(OUTPUT_PATH)
    
    # 加载所有数据
    all_data = load_data(DATA_CAPTIONS_PATH)
    total_items = len(all_data)
    print(f"共加载{total_items}项数据")
    
    # 过滤掉已处理的项
    filtered_data = []
    for item in all_data:
        if 'img_path' in item and item['img_path'] not in processed_img_paths:
            filtered_data.append(item)
    # 测试前200条
    # filtered_data = filtered_data[:200]
    total_processed = len(existing_results)
    total_to_process = len(filtered_data)
    
    print(f"已处理{total_processed}项，还需处理{total_to_process}项")
    
    if total_to_process == 0:
        print("所有数据都已处理完成，无需继续")
        return
    
    # 创建文件锁，确保并发写入安全
    file_lock = threading.Lock()

    print(f"开始并发处理{total_to_process}项数据...")
    max_workers = 20 # 控制并发数，避免API限制
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {executor.submit(process_item, item, idx, total_to_process): idx 
                         for idx, item in enumerate(filtered_data)}
        
        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=total_to_process):
            index, result = future.result()
            if result:
                # 保存处理结果
                save_item(result, file_lock, OUTPUT_PATH)
                total_processed += 1
                print(f"已处理并保存: {total_processed}项")
    
    # 最终保存所有结果（确保完整性）
    # 重新加载所有数据，确保最新
    final_results = []
    if os.path.exists(OUTPUT_PATH):
        final_results = load_data(OUTPUT_PATH)
    
    print(f"所有数据处理完成！总共处理: {total_processed}项")
    print(f"结果已保存到: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()