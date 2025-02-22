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
        for name, param in self.transformer_encoder.named_parameters():
            assert torch.isfinite(param.data).any(), f"transformer_encoder params not finite!\n Parameter: {name}\n Weights/Biases:\n{param.data}\n"
            if param.grad is not None:
                assert torch.isfinite(param.grad).any(), f"transformer_encoder grad not finite!\n Parameter: {name}\n grad:\n{param.grad}\n"
        # assert torch.isfinite(x).any(), f"encoder outputs infinite! transformer_encoder:{self.transformer_encoder.named_parameters()}"
        x = self.fc(x)
        assert torch.isfinite(x).any(), "fc outputs infinite!"
        logits = self.layer_norm(x)  # 稳定训练
        assert torch.isfinite(logits).any(), "logits infinite!"
        masked_logits = logits.masked_fill(~mask, -1e8)
        max_logit = masked_logits.max(dim=-1, keepdim=True).values
        stable_logits = masked_logits - max_logit
        probs = torch.softmax(stable_logits, dim=-1)

        # 添加概率下限保护（关键修改）
        min_prob = 1e-8  # 根据需求调整
        probs = probs.clamp(min=min_prob)  # 确保合法动作概率不低于min_prob
        # 重新归一化概率
        probs = probs / probs.sum(dim=-1, keepdim=True)  # 保证概率和为1

        assert torch.isfinite(probs).any(), "probs infinite!"
        # assert torch.any(probs == 0.), f"probs has zeros!\n probs:{probs}"
        assert (probs >= 0).all(), "probs has negative values!\n probs:{probs}"
        assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1))), "概率和不为1!"
        return probs            # 形状为(batch_size, action_dim)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
