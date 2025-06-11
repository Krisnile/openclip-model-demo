from models.image_encoder import ImageEncoder
from models.text_encoder import TextEncoder
from utils.dataset import CLIPCIFAR10
from utils.losses import clip_loss
from torch.utils.data import DataLoader
import torch, torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_enc = ImageEncoder().to(device)
txt_enc = TextEncoder().to(device)
optimizer = torch.optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()), lr=1e-4)

ds = CLIPCIFAR10(split='train')
loader = DataLoader(ds, batch_size=64, shuffle=True)

for epoch in range(10):
    for img, ids, mask in loader:
        img, ids, mask = img.to(device), ids.to(device), mask.to(device)
        img_feat = F.normalize(img_enc(img), dim=-1)
        txt_feat = F.normalize(txt_enc(ids, mask), dim=-1)
        loss = clip_loss(img_feat, txt_feat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss = {loss.item():.4f}")