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
            
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            mean_entropy = entropy[:valid_len].mean().item()
            
            log_likelihoods = []
            ranks = []
            for j in range(1, valid_len):
                target_id = seq_ids[j].item()
                prob_dist = probs[j-1]
                log_likelihoods.append(torch.log(prob_dist[target_id] + 1e-9).item())
                sorted_indices = torch.argsort(prob_dist, descending=True)
                rank = (sorted_indices == target_id).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
                
            mean_ll = np.mean(log_likelihoods) if log_likelihoods else 0
            mean_log_rank = np.mean(np.log(ranks)) if ranks else 0
            
            top_10 = sum(1 for r in ranks if r <= 10) / len(ranks) if ranks else 0
            top_100 = sum(1 for r in ranks if 10 < r <= 100) / len(ranks) if ranks else 0
            top_1000 = sum(1 for r in ranks if 100 < r <= 1000) / len(ranks) if ranks else 0
            others = sum(1 for r in ranks if r > 1000) / len(ranks) if ranks else 0
                
            batch_stats.append([mean_ll, mean_log_rank, mean_entropy, top_10, top_100, top_1000, others])
            
        return np.array(batch_stats)
