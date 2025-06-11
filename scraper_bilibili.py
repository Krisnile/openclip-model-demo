import os
import time
import requests
from io import BytesIO
from PIL import Image

# 模拟获取Bilibili视频BVids的函数
def fetch_bilibili_video_bvids():
    """
    模拟获取一组Bilibili视频的BVid。
    在实际应用中，这些BVid可能来自Bilibili排行榜、用户关注列表、搜索结果或其他复杂的Bilibili API（也可以参考bilibili-API-collect）。
    """
    return [
        "BV1Bh411V7C7",  # 【花好月圆夜】当BGM响起，瞬间回到那个夏天！【古风新韵】
        "BV1Xb4y1R77z",  # 2024年最期待的游戏，没有之一！
        "BV1gV411b7j7",  # 动漫剪辑：那些让人泪目的瞬间
        "BV1rW411E73g",  # 【英雄联盟】S14赛季宣传片：龙年限定
        "BV1GJ411c7Sj"   # 【原神】4.5版本前瞻特别节目
    ]

# 获取Bilibili视频封面链接的函数
def fetch_bilibili_cover_posts(bvids):
    """
    通过Bilibili官方API（如 SocialSisterYi/bilibili-API-collect 中记载的 /x/web-interface/view 接口）
    获取视频的标题和封面图片URL。
    """
    posts = []
    # 模拟浏览器请求头，这些头部信息有助于模拟真实用户请求，降低被封禁的风险。
    # Referer: 模拟从B站页面发出的请求
    # User-Agent: 模拟主流浏览器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://www.bilibili.com/'
    }

    for bvid in bvids:
        # 使用 SocialSisterYi/bilibili-API-collect 中记载的 "获取视频详细信息" 接口
        api_url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
        print(f"尝试通过 API 获取 BVID {bvid} 的视频信息: {api_url}")
        try:
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()  # 检查HTTP请求是否成功，如果状态码不是200，则抛出异常
            data = response.json()

            if data and data['code'] == 0 and 'data' in data and 'pic' in data['data']:
                title = data['data']['title']
                cover_url = data['data']['pic']
                posts.append({
                    "title": title,
                    "url": cover_url,
                    "bvid": bvid  # 保存bvid，用于后续生成文件名
                })
                print(f"成功获取：标题='{title}', 封面URL='{cover_url}'")
            else:
                # 打印B站API返回的错误信息（如果有）
                print(f"无法获取 BVID {bvid} 的视频信息：{data.get('message', '未知错误或数据结构不符')}")
        except requests.exceptions.RequestException as e:
            print(f"请求 Bilibili API 错误：{api_url}，错误信息：{e}")
        except Exception as e:
            print(f"处理 Bilibili API 响应失败：{api_url}，错误信息：{e}")
        finally:
            time.sleep(1)  # 礼貌性地等待，避免请求过快，降低被限流的风险

    return posts

# 下载并保存图片的函数
def download_image(image_source, save_path):
    """
    从给定的URL下载图片并保存到指定路径。
    此函数也使用模拟浏览器请求头，因为Bilibili的图片CDN可能也会检查Referer。
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9',
            'Connection': 'keep-alive',
            'Referer': 'https://www.bilibili.com/'  # 图片通常托管在Bilibili的CDN上，需要B站Referer
        }
        response = requests.get(image_source, timeout=10, headers=headers)
        response.raise_for_status()  # 检查HTTP请求是否成功
        image = Image.open(BytesIO(response.content)).convert("RGB") # 将图片统一转换为RGB模式，兼容性更好
        image.save(save_path)
        print(f"保存成功：{save_path}")
    except requests.exceptions.RequestException as e:
        print(f"请求图片资源错误：{image_source}，错误信息：{e}")
    except Exception as e:
        print(f"处理图片失败：{image_source}，错误信息：{e}")
    finally:
        time.sleep(1) # 增加下载之间的延迟，进一步避免触发限速

# 主执行逻辑
if __name__ == "__main__":
    output_dir = "bilibili_video_covers"  # 设置输出目录名称
    os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在则创建

    # 1. 获取需要爬取封面的Bilibili视频BVids列表
    bvids_to_crawl = fetch_bilibili_video_bvids()

    # 2. 根据BVids，通过Bilibili API获取视频的标题和封面URL
    video_posts = fetch_bilibili_cover_posts(bvids_to_crawl)

    # 3. 遍历并下载每个视频的封面
    for post in video_posts:
        title = post['title']
        url = post['url']
        bvid = post['bvid']

        # 根据BVID和标题生成安全的文件名
        # 移除非字母数字字符，替换为空格，然后去除首尾空格并替换内部多个空格为单个下划线
        safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_', '.') else "_" for c in title).strip()
        safe_title = "_".join(safe_title.split()) # 将多个空格替换为单个下划线
        
        # 限制文件名的长度，避免文件名过长导致系统问题
        if len(safe_title) > 100: # 稍微放宽长度，但仍控制
            safe_title = safe_title[:100] + "..."

        # 最终文件名格式：BVID_视频标题.jpg
        filename = f"{bvid}_{safe_title}.jpg"
        save_path = os.path.join(output_dir, filename)
        
        download_image(url, save_path)

    print("\n所有Bilibili视频封面下载任务完成！")