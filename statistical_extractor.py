import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import softmax

class StatisticalExtractor:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base-mlm")
        self.model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base-mlm").to(self.device)
        self.model.eval()

    def extract_batch(self, codes):
        inputs = self.tokenizer(codes, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            # Vectorized softmax across the entire batch simultaneously
            probs = softmax(logits, dim=-1)

        batch_stats = []
        for i in range(logits.size(0)):
            valid_len = attention_mask[i].sum().item()
            
            if valid_len > 2:
                inner_probs = probs[i, 1:valid_len-1]
                inner_ids = input_ids[i, 1:valid_len-1]
                
                entropy = -(inner_probs * torch.log(inner_probs + 1e-9)).sum(dim=-1).mean()
                
                target_probs = inner_probs.gather(1, inner_ids.unsqueeze(1)).squeeze(1)
                mean_ll = torch.log(target_probs + 1e-9).mean()
                
                # O(V) lossless rank calculation replacing O(V log V) argsort
                ranks = (inner_probs > target_probs.unsqueeze(1)).sum(dim=-1) + 1
                ranks_float = ranks.float()
                
                mean_log_rank = torch.log(ranks_float).mean()
                total_tokens = float(ranks.size(0))
                
                top_10 = (ranks <= 10).sum().float() / total_tokens
                top_100 = ((ranks > 10) & (ranks <= 100)).sum().float() / total_tokens
                top_1000 = ((ranks > 100) & (ranks <= 1000)).sum().float() / total_tokens
                others = (ranks > 1000).sum().float() / total_tokens
                
                # Single sync to CPU per sequence
                batch_stats.append([
                    mean_ll.item(), mean_log_rank.item(), entropy.item(), 
                    top_10.item(), top_100.item(), top_1000.item(), others.item()
                ])
            else:
                batch_stats.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
        return np.array(batch_stats, dtype=np.float32)
