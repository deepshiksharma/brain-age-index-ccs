from torch import nn

class LinearBA(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1, bias=True)
    def forward(self, x):
        return self.linear(x).squeeze(-1)
