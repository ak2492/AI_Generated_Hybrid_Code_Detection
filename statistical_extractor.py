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
        
        with torch.no_grad():
            logits = self.model(**inputs).logits

        batch_stats = []
        for i in range(logits.size(0)):
            seq_logits = logits[i]
            seq_ids = input_ids[i]
            valid_len = inputs["attention_mask"][i].sum().item()
            probs = softmax(seq_logits, dim=-1)
            
            # Vectorized Entropy
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            mean_entropy = entropy[:valid_len].mean().item()
            
            if valid_len > 1:
                valid_probs = probs[:valid_len-1]
                target_ids = seq_ids[1:valid_len]
                
                # Vectorized Log-Likelihood
                target_probs = valid_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
                log_likelihoods = torch.log(target_probs + 1e-9).cpu().numpy()
                mean_ll = np.mean(log_likelihoods)
                
                # Vectorized Rank Calculation
                sorted_indices = torch.argsort(valid_probs, dim=1, descending=True)
                ranks = (sorted_indices == target_ids.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
                ranks = ranks.cpu().numpy()
                
                mean_log_rank = np.mean(np.log(ranks))
                top_10 = np.sum(ranks <= 10) / len(ranks)
                top_100 = np.sum((ranks > 10) & (ranks <= 100)) / len(ranks)
                top_1000 = np.sum((ranks > 100) & (ranks <= 1000)) / len(ranks)
                others = np.sum(ranks > 1000) / len(ranks)
            else:
                mean_ll, mean_log_rank, top_10, top_100, top_1000, others = 0, 0, 0, 0, 0, 0
                
            batch_stats.append([mean_ll, mean_log_rank, mean_entropy, top_10, top_100, top_1000, others])
            
        return np.array(batch_stats)
