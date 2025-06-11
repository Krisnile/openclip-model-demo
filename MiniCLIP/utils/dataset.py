from torchvision import datasets, transforms
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer

class CLIPCIFAR10(Dataset):
    def __init__(self, split='train'):
        self.dataset = datasets.CIFAR10('./data', train=(split=='train'), download=True,
                                        transform=transforms.ToTensor())
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.labels = self.dataset.classes

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        text = f"A photo of a {self.labels[label]}"
        text_tok = self.tokenizer(text, return_tensors="pt", padding='max_length',
                                  truncation=True, max_length=32)
        return img, text_tok['input_ids'].squeeze(0), text_tok['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.dataset)