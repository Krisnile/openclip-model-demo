from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.proj = nn.Linear(768, output_dim)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.proj(out)