import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

        # 添加归一化层（可选但推荐）
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)  # 稳定训练
        logits = self.fc2(x)    # 输出原始Logits，而非概率
        masked_logits = logits.masked_fill(~mask, -1e2)
        max_logit = masked_logits.max(dim=-1, keepdim=True).values
        stable_logits = masked_logits - max_logit
        probs = torch.softmax(stable_logits, dim=-1)
        return probs            # 形状为(batch_size, action_dim)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
