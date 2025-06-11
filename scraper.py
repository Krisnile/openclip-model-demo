import os
import time
import requests
from io import BytesIO
from PIL import Image

# 模拟获取图片链接的函数
def fetch_image_posts():
    return [
        {
            "title": "Apollo 17 Lunar Rover",
            "url": "https://upload.wikimedia.org/wikipedia/commons/5/5f/Apollo_17_Lunar_Rover.jpg"
        },
        {
            "title": "Mona Lisa",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/640px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
        },
        {
            "title": "Python Logo",
            "url": "https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg"
        },
        {
            "title": "Eiffel Tower",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/800px-Tour_Eiffel_Wikimedia_Commons.jpg"
        },
        {
            "title": "Aurora Borealis",
            "url": "https://upload.wikimedia.org/wikipedia/commons/9/99/Aurora_Borealis_above_Lyngenfjord.jpg"
        }
    ]

# 下载并保存图片的函数
def download_image(image_source, save_path):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://commons.wikimedia.org/'
        }
        response = requests.get(image_source, timeout=10, headers=headers)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.save(save_path)
        print(f"保存成功：{save_path}")
    except requests.exceptions.RequestException as e:
        print(f"请求错误：{image_source}，错误信息：{e}")
    except Exception as e:
        print(f"处理图片失败：{image_source}，错误信息：{e}")
    finally:
        time.sleep(1)

# 主执行逻辑
if __name__ == "__main__":
    output_dir = "downloaded_images"
    os.makedirs(output_dir, exist_ok=True)

    posts = fetch_image_posts()
    for idx, post in enumerate(posts):
        title = post['title']
        url = post['url']
        # 根据标题生成安全的文件名
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        filename = f"{idx+1:02d}_{safe_title}.jpg"
        save_path = os.path.join(output_dir, filename)
        download_image(url, save_path)
