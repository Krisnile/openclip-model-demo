import torch
import torch.nn.functional as F

def clip_loss(image_features, text_features, temperature=0.07):
    logits_per_image = image_features @ text_features.T / temperature
    logits_per_text = logits_per_image.T
    labels = torch.arange(image_features.size(0)).to(image_features.device)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2