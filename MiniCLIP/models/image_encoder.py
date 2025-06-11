import torchvision.models as models
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.backbone(x).squeeze()
        x = self.proj(x)
        return x