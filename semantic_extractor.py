import torch
from transformers import AutoTokenizer, T5EncoderModel

class SemanticExtractor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m")
        self.model = T5EncoderModel.from_pretrained("Salesforce/codet5p-220m").to(self.device)
        self.model.eval()

    def extract_batch(self, codes):
        inputs = self.tokenizer(codes, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Reverted to literal [CLS] (first token) extraction to match Paper Sec. 3.1.3
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
