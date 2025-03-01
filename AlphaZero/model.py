import torch
import torch.nn as nn
import torch.nn.functional as F

class GomokuNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(GomokuNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(
            args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024
        )
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # you can add residual to the network
        #                                                           s: batch_size x board_x x board_y
        s = s.view(
            -1, 1, self.board_x, self.board_y
        )  # batch_size x 1 x board_x x board_y
        s = F.relu(
            self.bn1(self.conv1(s))
        )  # batch_size x num_channels x board_x x board_y
        s = F.relu(
            self.bn2(self.conv2(s))
        )  # batch_size x num_channels x board_x x board_y
        s = F.relu(
            self.bn3(self.conv3(s))
        )  # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(
            self.bn4(self.conv4(s))
        )  # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.args.dropout,
            training=self.training,
        )  # batch_size x 1024
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.args.dropout,
            training=self.training,
        )  # batch_size x 512

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
class MyEncoderNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_size = game.getActionSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(MyEncoderNet, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.board_size, nhead=self.args.num_heads, 
                                                        dim_feedforward=self.args.hidden_dim, layer_norm_eps=1e-5, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.args.num_blocks)
        self.fc3 = nn.Linear(self.board_size, self.action_size)
        self.fc4 = nn.Linear(self.board_size, 1)

        # 添加归一化层（可选但推荐）
        self.batch_norm = torch.nn.BatchNorm1d(self.board_size)

    def forward(self, s):
        # you can add residual to the network
        #                                                           s: batch_size x board_x x board_y
        s = s.view(
            -1, self.board_size
        )  # batch_size x board_size
        s = self.transformer_encoder(s)
        for name, param in self.transformer_encoder.named_parameters():
            assert torch.isfinite(param.data).any(), f"transformer_encoder params not finite!\n Parameter: {name}\n Weights/Biases:\n{param.data}\n"
            if param.grad is not None:
                assert torch.isfinite(param.grad).any(), f"transformer_encoder grad not finite!\n Parameter: {name}\n grad:\n{param.grad}\n"
        
        # logits = self.layer_norm(s)  # 稳定训练
        # assert torch.isfinite(logits).any(), "logits infinite!"
        # max_logit = logits.max(dim=-1, keepdim=True).values
        # stable_logits = logits - max_logit
        # probs = torch.softmax(stable_logits, dim=-1)

        s = F.dropout(
            F.relu(self.batch_norm(s)),
            p=self.args.dropout,
            training=self.training,
        )  # batch_size x 512
        assert torch.isfinite(s).any(), "fc outputs infinite!"

        pi = self.fc3(s)  # batch_size x action_size
        v = self.fc4(s)  # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)