import os
import sys

# 添加项目根目录到Python路径
sys.path.append('/home/wangrui/code/MLLM4Text-ReID-main')

from reid_demo import ReIDDemo

def find_similar_images(query_image_path, text_query=None, top_k=5, img_weight=0.5, text_weight=0.5):
    """
    查找与给定图像和文本描述相似的图像
    
    参数:
    query_image_path: 字符串，查询图像的路径（相对于data/RSTPReid/imgs目录或绝对路径）
    text_query: 字符串，自然语言描述（可选）
    top_k: 整数，返回结果的数量
    img_weight: 浮点数，图像相似度权重
    text_weight: 浮点数，文本相似度权重
    
    返回:
    列表，包含(top_k)个元组，每个元组为(img_path, person_id, similarity_score)
    """
    # 配置和模型路径
    config_path = '/home/wangrui/code/MLLM4Text-ReID-main/logs/RSTPReid_1023/20251023_234421_finetune/configs.yaml'
    model_path = '/home/wangrui/code/MLLM4Text-ReID-main/checkpoint/best2.pth'
    
    try:
        # 创建演示实例
        demo = ReIDDemo(config_path, model_path)
        
        # 搜索相似图像，使用与reid_demo.py中相同的参数
        results = demo.search_similar_images(
            query_image_path, 
            text_query=text_query, 
            top_k=top_k,
            img_weight=img_weight,
            text_weight=text_weight
        )
        
        return results
    except Exception as e:
        print(f"错误: {e}")
        return []

# 命令行测试接口
def main():
    """命令行接口，支持更多参数"""
    # 从命令行参数获取输入
    if len(sys.argv) >= 3:
        query_image = sys.argv[1]
        text_query = sys.argv[2]
        top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        img_weight = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        text_weight = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    else:
        # 默认测试用例
        query_image = "3902_c5_0005.jpg"
        text_query = "A man with dark hair wearing a black overcoat and black pants"
        top_k = 5
        img_weight = 0.5
        text_weight = 0.5
    
    print(f"查询图像: {query_image}")
    print(f"查询文本: {text_query}")
    print(f"参数设置 - top_k: {top_k}, 图像权重: {img_weight}, 文本权重: {text_weight}")
    
    # 查找相似图像
    results = find_similar_images(query_image, text_query, top_k, img_weight, text_weight)
    
    # 打印结果
    print("\n相似图像结果:")
    for i, (img_path, person_id, score) in enumerate(results):
        print(f"{i+1}. 图像: {img_path}, 人物ID: {person_id}, 相似度: {score:.4f}")
    
    # 返回最佳匹配的完整路径
    if results:
        best_match_path = results[0][0]
        # 确保路径正确
        if not os.path.isabs(best_match_path):
            full_path = os.path.join('/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs', best_match_path)
        else:
            full_path = best_match_path
        print(f"\n最佳匹配图像的完整路径: {full_path}")
        return full_path
    
    return None

if __name__ == '__main__':
    main()