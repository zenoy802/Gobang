import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
from tqdm import tqdm
import numpy as np
import wandb

class NNetWrapper:
    def __init__(self, game, args):
        self.nnet = GomokuNNet(game, args).to(args.device)
        # self.nnet = MyEncoderNet(game, args).to(args.device)
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        self.device = args.device
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.max_lr)
        
        # 1cycle learning rate parameters
        self.total_steps = args.numIters * args.epochs * (args.maxlenOfQueue // args.batch_size)
        self.current_step = 0

    def get_learning_rate(self):
        """Implement 1cycle learning rate strategy"""
        if self.current_step >= self.total_steps:
            return self.args.min_lr
        
        # Divide the total steps into two phases
        half_cycle = self.total_steps // 2
        
        if self.current_step <= half_cycle:
            # First phase: increase from min_lr to max_lr
            phase = self.current_step / half_cycle
            lr = self.args.min_lr + (self.args.max_lr - self.args.min_lr) * phase
        else:
            # Second phase: decrease from max_lr to min_lr
            phase = (self.current_step - half_cycle) / half_cycle
            lr = self.args.max_lr - (self.args.max_lr - self.args.min_lr) * phase
        
        return lr

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        for epoch in range(self.args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / self.args.batch_size)

            t = tqdm(range(batch_count), desc="Training Net")
            for _ in t:
                # Update learning rate
                lr = self.get_learning_rate()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                self.current_step += 1

                sample_ids = np.random.randint(len(examples), size=self.args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float32)).to(self.device)
                # TODO: why it is target?
                target_pis = torch.FloatTensor(np.array(pis)).to(self.device)
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float32)).to(self.device)


                # compute output
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses, lr=f"{lr:.1e}")

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Add gradient clipping
                if self.args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), self.args.grad_clip)
                
                self.optimizer.step()

                if getattr(self.args, 'wandb', False):
                    wandb.log({
                        'learning_rate': lr,
                        'policy_loss': l_pi.item(),
                        'value_loss': l_v.item(),
                        'total_loss': total_loss.item(),
                        'current_step': self.current_step,
                    })

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        # start = time.time()

        # preparing input
        board = torch.FloatTensor(board.astype(np.float32)).to(self.device)
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        folder = folder.rstrip('/')
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise ValueError("No model in path {}".format(filepath))
        map_location = None if self.args.cuda else "cpu"
        if self.args.mps:
            map_location = "mps"
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        self.nnet.load_state_dict(checkpoint["state_dict"])


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f"{self.avg:.2e}"

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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