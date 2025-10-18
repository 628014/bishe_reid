from openai import OpenAI
import os
import base64
from prompt import prompt

from dashscope import MultiModalConversation
import dashscope 


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_qwen_call_base64():
    base64_image = encode_image("/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/0000_c1_0004.jpg")

    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为：https://dashscope-intl.aliyuncs.com/compatible-mode/v1
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


    completion = client.chat.completions.create(
        model="qwen3-vl-plus", # 此处以qwen3-vl-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/models
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        # 需要注意，传入Base64，图像格式（即image/{format}）需要与支持的图片列表中的Content Type保持一致。"f"是字符串格式化的方法。
                        # PNG图像：  f"data:image/png;base64,{base64_image}"
                        # JPEG图像： f"data:image/jpeg;base64,{base64_image}"
                        # WEBP图像： f"data:image/webp;base64,{base64_image}"
                        # "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, 
                        "image_url": {"url": image_path},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    print(completion.choices[0].message.content)

def get_qwen_call_local(local_path, custom_prompt=None):
    """调用通义千问VL模型处理本地图片
    Args:
        local_path: 本地图片的绝对路径
        custom_prompt: 自定义prompt，如果为None则使用默认prompt
        
    Returns:
        str: 模型返回的文本内容
    """
    # 若使用新加坡地域的模型，请取消下列注释
    # dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

    image_path = f"file://{local_path}"
    # 使用自定义prompt或默认prompt
    current_prompt = custom_prompt if custom_prompt else prompt
    messages = [
                    {'role':'user',
                    'content': [{'image': image_path},
                                {'text': current_prompt }]}]
    try:
        response = MultiModalConversation.call(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
            # 新加坡和北京地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/models
            api_key=os.getenv('DASHSCOPE_API_KEY'),
            model='qwen3-vl-plus',  # 此处以qwen3-vl-plus为例，可按需更换模型名称
            messages=messages)
        return response["output"]["choices"][0]["message"]["content"][0]["text"]
    except Exception as e:
        print(f"调用通义千问API时出错: {e}")
        return None



if __name__ == "__main__":
    # 示例调用
    local_path = "/home/wangrui/code/MLLM4Text-ReID-main/data/RSTPReid/imgs/0000_c1_0004.jpg"
    result = get_qwen_call_local(local_path)
    print(result)


