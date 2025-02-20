import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, num_heads):
        super(PolicyNet, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=state_dim, nhead=num_heads, 
                                                        dim_feedforward=hidden_dim, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(state_dim, action_dim)

        # 添加归一化层（可选但推荐）
        self.layer_norm = torch.nn.LayerNorm(action_dim)

    def forward(self, x, mask):
        x = self.transformer_encoder(x)
        assert torch.isfinite(x).any(), "encoder outputs infinite!"
        x = self.fc(x)
        assert torch.isfinite(x).any(), "fc outputs infinite!"
        logits = self.layer_norm(x)  # 稳定训练
        assert torch.isfinite(logits).any(), "logits infinite!"
        masked_logits = logits.masked_fill(~mask, -1e3)
        max_logit = masked_logits.max(dim=-1, keepdim=True).values
        stable_logits = masked_logits - max_logit
        probs = torch.softmax(stable_logits, dim=-1)
        assert torch.isfinite(probs).any(), "probs infinite!"
        return probs            # 形状为(batch_size, action_dim)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
