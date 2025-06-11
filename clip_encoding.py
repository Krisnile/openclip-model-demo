# clip_encoding.py (Modified for open_clip)
import torch
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time
import os
import open_clip

# Load open_clip model and processor once globally
# Using 'RN50' as the model name and 'openai' as the pretrained dataset
# You can explore other models and datasets with open_clip.list_pretrained()
model_name = 'RN50'
pretrained_dataset = 'openai' # Or 'laion400m_e32', 'laion2b_s34b_b79k', etc. for different models/checkpoints

# Load the model and its associated preprocessing transform
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_dataset)
# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval() # Set model to evaluation mode

# We'll use the tokenizer from open_clip
tokenizer = open_clip.get_tokenizer(model_name)


def load_image(image_source):
    """
    Loads an image from either a URL or a local file path.
    Args:
        image_source (str): The URL or local file path of the image.
    Returns:
        PIL.Image.Image: The loaded image as a PIL Image object, or None if failed.
    """
    if os.path.exists(image_source):
        # It's a local file path
        try:
            image = Image.open(image_source).convert("RGB")
            return image
        except Exception as e:
            print(f"Error loading local image from {image_source}: {e}")
            return None
        finally:
            time.sleep(0.01) # 短暂延迟，即使是本地文件也保持良好习惯
    else:
        # Assume it's a URL (original behavior if you still want to support it)
        try:
            import requests # Only import requests if actually needed for URLs
            headers = {
                'User-Agent': 'MyCLIPImageScraper/1.0 (contact: your_email@example.com)' 
            }
            response = requests.get(image_source, timeout=10, headers=headers)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {image_source}: {e}")
            return None
        except Exception as e:
            print(f"Error processing image from {image_source}: {e}")
            return None
        finally:
            time.sleep(1) # 如果是URL下载，保持较长延迟

def encode_image_with_clip(image):
    """Encodes a PIL Image into an open_clip feature vector."""
    if image is None:
        return None
    # Preprocess the image using the transform loaded with the model
    image_input = preprocess(image).unsqueeze(0).to(device) # Add batch dimension and move to device
    
    with torch.no_grad():
        features = model.encode_image(image_input)
    
    # Normalize features if your downstream similarity calculation assumes it
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy() # Return as numpy array

def encode_text_with_clip(text):
    """Encodes a text string into an open_clip feature vector."""
    # Tokenize the text using open_clip's tokenizer
    text_input = tokenizer(text).to(device)
    
    with torch.no_grad():
        features = model.encode_text(text_input)
    
    # Normalize features
    features = features / features.norm(p=2, dim=-1, keepdim=True)
    return features.squeeze().cpu().numpy() # Return as numpy array

if __name__ == "__main__":
    print(f"Using open_clip model: {model_name}, pretrained on: {pretrained_dataset}")
    print(f"Model device: {device}")

    # 修改测试用例以使用本地图片
    test_img_path = "local_images/000000015497.jpg" # 确保这个路径存在
    if os.path.exists(test_img_path):
        image = load_image(test_img_path)
        if image:
            features = encode_image_with_clip(image)
            print(f"Encoded local image features shape: {features.shape}")
            print(f"Example features: {features[:5]}")
        else:
            print(f"Failed to load test local image: {test_img_path}")
    else:
        print(f"Test local image not found: {test_img_path}. Please create it.")
        
    test_text = "a dog walking on the beach"
    text_features = encode_text_with_clip(test_text)
    print(f"Encoded text features shape: {text_features.shape}")
    print(f"Example text features: {text_features[:5]}")