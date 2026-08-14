import torch.nn as nn

class HybridCodeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # 813 Dimensions: Authorship (38) + Statistical (7) + Semantic (768)
        self.classifier = nn.Sequential(
            nn.Linear(813, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.classifier(x).squeeze()
