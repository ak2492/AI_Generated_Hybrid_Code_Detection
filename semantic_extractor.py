import torch
from transformers import AutoTokenizer, T5EncoderModel

class SemanticExtractor:
    def __init__(self, device):
        self.device = device
        # Overriding the config bug and explicitly loading the Encoder-only configuration
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-220m", extra_special_tokens=[])
        self.model = T5EncoderModel.from_pretrained("Salesforce/codet5p-220m").to(self.device)
        self.model.eval()

    def extract_batch(self, codes):
        inputs = self.tokenizer(codes, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            # Since we loaded T5EncoderModel, it no longer looks for decoder inputs
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()
