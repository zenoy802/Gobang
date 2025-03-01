import logging
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from collections import deque
from random import shuffle
import wandb
import yaml

import game

from model import NNetWrapper

import timeit
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MCTS:
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnetWrapper, args):
        self.game = game
        self.nnetWrapper = nnetWrapper
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, canonicalBoard, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        startTime = timeit.default_timer()
        for _ in range(self.args.numMCTSSims):
            self.recursion_search(canonicalBoard)
            # self.loop_search(canonicalBoard)
        endTime = timeit.default_timer()
        print(f"search time {endTime - startTime}")

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [
            self.Nsa[(s, a)] if (s, a) in self.Nsa else 0
            for a in range(self.game.getActionSize())
        ]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1.0 / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def loop_search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS using a loop instead of recursion.
        """
        current_board = canonicalBoard
        path = []

        while True:
            s = self.game.stringRepresentation(current_board)

            if s not in self.Es:
                self.Es[s] = self.game.getGameEnded(current_board, 1)
            if self.Es[s] is not None:
                v = self.Es[s]
                break

            if s not in self.Ps:
                # leaf node
                self.Ps[s], v = self.nnetWrapper.predict(current_board)
                valids = self.game.getValidMoves(current_board, 1)
                self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
                sum_Ps_s = np.sum(self.Ps[s])
                if sum_Ps_s > 0:
                    self.Ps[s] /= sum_Ps_s  # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[s] = self.Ps[s] + valids
                    self.Ps[s] /= np.sum(self.Ps[s])

                self.Vs[s] = valids
                self.Ns[s] = 0
                break

            valids = self.Vs[s]
            cur_best = -float("inf")
            best_act = -1

            # pick the action with the highest upper confidence bound
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    u = self.Qsa.get((s, a), 0) + self.args.cpuct * self.Ps[s][
                        a
                    ] * math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0))

                    if u > cur_best:
                        cur_best = u
                        best_act = a

            a = best_act
            path.append((s, a))
            next_s, next_player = self.game.getNextState(current_board, 1, a)
            current_board = self.game.getCanonicalForm(next_s, next_player)

        # Backpropagation phase
        for s, a in reversed(path):
            v = -v 

            if (s, a) in self.Qsa:
                self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                    self.Nsa[(s, a)] + 1
                )
                self.Nsa[(s, a)] += 1

            else:
                self.Qsa[(s, a)] = v
                self.Nsa[(s, a)] = 1

            self.Ns[s] += 1
        
        return v

    def recursion_search(self, canonicalBoard):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: Since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the value of the current canonicalBoard
        """

        s = self.game.stringRepresentation(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        if self.Es[s] is not None:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnetWrapper.predict(canonicalBoard)
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        valids = self.Vs[s]
        cur_best = -float("inf")
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.game.getActionSize()):
            if valids[a]:
                u = self.Qsa.get((s, a), 0) + self.args.cpuct * self.Ps[s][
                    a
                ] * math.sqrt(self.Ns[s]) / (1 + self.Nsa.get((s, a), 0))

                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        # SIMD (vectorized) implementation of PUTC values
        # action_size = self.game.getActionSize()
        # Q_values = np.zeros(action_size, dtype=np.float32)
        # N_visits = np.zeros(action_size, dtype=np.float32)
        # for a in range(action_size):
        #     Q_values[a] = float(self.Qsa.get((s, a), 0))
        #     N_visits[a] = float(self.Nsa.get((s, a), 0))
        # U = Q_values + self.args.cpuct * self.Ps[s] * np.sqrt(self.Ns[s]) / (1 + N_visits)
        # # masking invalid moves
        # U = np.where(valids, U, -np.inf)
        # best_act = np.argmax(U)

        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = -self.recursion_search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (
                self.Nsa[(s, a)] + 1
            )
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v


class TrainExamplesGenerator:

    def __init__(self, game, nnetWrapper, args):
        self.game = game
        self.nnetWrapper = nnetWrapper
        self.args = args
        self.mcts = MCTS(self.game, self.nnetWrapper, self.args)
    
    def __getstate__(self):
        """Return state values to be pickled"""
        state = self.__dict__.copy()
        del state['nnet']
        del state['mcts']
        return state
    
    def __setstate__(self, state, model_filename=None):
        """Restore state from the unpickled state values"""
        self.__dict__.update(state)
        if not model_filename:
            self.nnetWrapper = NNetWrapper(self.game, self.args)
            self.mcts = MCTS(self.game, self.nnetWrapper, self.args)
        else:
            self.nnetWrapper = NNetWrapper(self.game, self.args)
            self.nnetWrapper.load_checkpoint(folder=self.args.checkpoint, filename=model_filename)
            self.mcts = MCTS(self.game, self.nnetWrapper, self.args)

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, pi, v)
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(
                board, self.curPlayer, action
            )

            r = self.game.getGameEnded(board, self.curPlayer)

            if r is not None:
                # r * (1 if self.curPlayer == x[1] else -1) means 1 for winner, -1 for loser, 0 for draw.
                # 一旦胜利则每步奖励都为1，失败则每步奖励都为-1
                return [
                    (x[0], x[2], r * (1 if self.curPlayer == x[1] else -1))
                    for x in trainExamples
                ]
    
    def generate_train_examples_sync(self, num_episodes, worker_id):
        # examples of the iteration
        iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
        for _ in tqdm(range(num_episodes), desc=f"Self Play on worker {worker_id}"):
            iterationTrainExamples += self.executeEpisode()
        return iterationTrainExamples


class SelfPlay:
    """
    This class executes the self-play + learning.
    """

    def __init__(self, game, nnetWrapper, args):
        self.game = game
        self.nnetWrapper = nnetWrapper
        self.pnetWrapper = self.nnetWrapper.__class__(self.game, args)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnetWrapper, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations

    def generate_train_examples(self):
        generator = TrainExamplesGenerator(self.game, self.nnetWrapper, self.args)
        iterationTrainExamples = generator.generate_train_examples_sync(self.args.numEps, 0)
        return iterationTrainExamples
    
    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f"Starting Iter #{i} ...")
            self.mcts = MCTS(self.game, self.nnetWrapper, self.args)  # reset search tree
            iterationTrainExamples = self.generate_train_examples()

            # save the iteration examples to the history
            self.trainExamplesHistory.append(iterationTrainExamples)

            if (
                len(self.trainExamplesHistory)
                > self.args.numItersForTrainExamplesHistory
            ):
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}"
                )
                self.trainExamplesHistory.pop(0)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnetWrapper.save_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            self.pnet.load_checkpoint(
                folder=self.args.checkpoint, filename="temp.pth.tar"
            )
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnetWrapper.train(trainExamples)
            nmcts = MCTS(self.game, self.nnetWrapper, self.args)

            log.info("PITTING AGAINST PREVIOUS VERSION")
            arena = game.Arena(
                lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                lambda x: np.argmax(nmcts.getActionProb(x, temp=0)),
                self.game,
            )
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info("NEW/PREV WINS : %d / %d ; DRAWS : %d" % (nwins, pwins, draws))
            if (
                pwins + nwins == 0
                or float(nwins) / (pwins + nwins) < self.args.updateThreshold
            ):
                log.info("REJECTING NEW MODEL")
                self.nnetWrapper.load_checkpoint(
                    folder=self.args.checkpoint, filename="temp.pth.tar"
                )
            else:
                log.info("ACCEPTING NEW MODEL")
                self.nnetWrapper.save_checkpoint(
                    folder=self.args.checkpoint, filename="best.pth.tar"
                )


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert nested dict to dotdict
    args = dotdict({})
    
    # Training params
    args.epochs = config['training']['epochs']
    args.batch_size = config['training']['batch_size']
    args.numIters = config['training']['num_iterations']
    args.numEps = config['training']['num_episodes']
    args.maxlenOfQueue = config['training']['max_queue_length']
    args.numItersForTrainExamplesHistory = config['training']['num_iters_history']
    args.updateThreshold = config['training']['update_threshold']
    args.arenaCompare = config['training']['arena_compare']
    args.tempThreshold = config['training']['temp_threshold']
    
    # Network params
    args.num_channels = config['network']['num_channels']
    args.dropout = config['network']['dropout']
    args.min_lr = config['network']['learning_rate']['min']
    args.max_lr = config['network']['learning_rate']['max']
    args.grad_clip = config['network']['grad_clip']
    # encoder params
    args.num_heads = config['encoder']['num_heads']
    args.hidden_dim = config['encoder']['hidden_dim']
    args.num_blocks = config['encoder']['num_blocks']
    
    # MCTS params
    args.numMCTSSims = config['mcts']['num_sims']
    args.cpuct = config['mcts']['cpuct']
    
    # Game params
    args.board_size = config['game']['board_size']
    
    # System params
    args.cuda = torch.cuda.is_available()
    args.mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if args.cuda:
        args.device = torch.device("cuda")
    elif args.mps:
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
    args.checkpoint = config['system']['checkpoint_dir']
    args.load_model = config['system']['load_model']
    args.load_folder_file = tuple(config['system']['load_folder_file'])
    
    return args


def print_config(args):
    """Pretty print the configuration"""
    print("\n=== Configuration ===")
    print("Training Parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Number of Iterations: {args.numIters}")
    print(f"  Episodes per Iteration: {args.numEps}")
    print(f"  Max Queue Length: {args.maxlenOfQueue}")
    print(f"  Training History Length: {args.numItersForTrainExamplesHistory}")
    print(f"  Update Threshold: {args.updateThreshold}")
    print(f"  Arena Compare Games: {args.arenaCompare}")
    print(f"  Temperature Threshold: {args.tempThreshold}")
    
    print("\nNetwork Parameters:")
    print(f"  Number of Channels: {args.num_channels}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning Rate Range: {args.min_lr} - {args.max_lr}")
    print(f"  Gradient Clip: {args.grad_clip}")
    
    print("\nMCTS Parameters:")
    print(f"  MCTS Simulations: {args.numMCTSSims}")
    print(f"  CPUCT: {args.cpuct}")
    
    print("\nGame Parameters:")
    print(f"  Board Size: {args.board_size}")
    
    print("\nSystem Parameters:")
    print(f"  CUDA Enabled: {args.cuda}")
    print(f"  MPS Enabled: {args.mps}")
    print(f"  Using Device: {args.device}")
    print(f"  Checkpoint Directory: {args.checkpoint}")
    print(f"  Load Model: {args.load_model}")
    print(f"  Load Path: {args.load_folder_file}")
    print("==================\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml", help="Path to config file")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--board_size", type=int, default=9)
    # play arguments
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--round", type=int, default=2)
    parser.add_argument(
        "--player1",
        type=str,
        default="human",
        choices=["human", "random", "greedy", "alphazero"],
    )
    parser.add_argument(
        "--player2",
        type=str,
        default="alphazero",
        choices=["human", "random", "greedy", "alphazero"],
    )
    parser.add_argument("--ckpt_file", type=str, default="best.pth.tar")
    parser.add_argument("--wandb", action="store_true", help="Use wandb to record the training process")
    parser.add_argument("--wandb_project", type=str, default="alphazero-gomoku", help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="wandb entity name")
    parser.add_argument("--wandb_id", type=str, default=None)
    
    args_input = vars(parser.parse_args())
    
    # Load config and override with command line arguments
    args = load_config(args_input['config'])
    for k, v in args_input.items():
        if k != 'config':
            args[k] = v
    
    # Add this line to print configuration
    print_config(args)
    
    g = game.GomokuGame(args.board_size)

    if args.train:
        nnet = NNetWrapper(g, args)
        if args.load_model:
            log.info(
                'Loading checkpoint "%s/%s"...',
                args.load_folder_file[0],
                args.load_folder_file[1],
            )
            nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
            # Initialize wandb
            if args.wandb:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    id=args.wandb_id,
                    config={
                        "board_size": args.board_size,
                        "num_iterations": args.numIters,
                        "num_episodes": args.numEps,
                        "num_mcts_sims": args.numMCTSSims,
                        "batch_size": args.batch_size,
                        "num_channels": args.num_channels,
                        "learning_rate_min": args.min_lr,
                        "learning_rate_max": args.max_lr,
                        "grad_clip": args.grad_clip,
                        "epochs": args.epochs,
                        "dropout": args.dropout,
                    },
                    resume="allow"
                )
        else:
            # Initialize wandb
            if args.wandb:
                wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    config={
                        "board_size": args.board_size,
                        "num_iterations": args.numIters,
                        "num_episodes": args.numEps,
                        "num_mcts_sims": args.numMCTSSims,
                        "batch_size": args.batch_size,
                        "num_channels": args.num_channels,
                        "learning_rate_min": args.min_lr,
                        "learning_rate_max": args.max_lr,
                        "grad_clip": args.grad_clip,
                        "epochs": args.epochs,
                        "dropout": args.dropout,
                    },
                    resume="allow"
                )

        log.info("Loading the SelfCoach...")
        s = SelfPlay(g, nnet, args)

        log.info("Starting the learning process 🎉")
        s.learn()

    if args.play:
        def getPlayFunc(name):
            if name == "human":
                return game.HumanGomokuPlayer(g).play
            elif name == "random":
                return game.RandomGomokuPlayer(g).play
            elif name == "greedy":
                return game.GreedyGomokuPlayer(g).play
            elif name == "alphazero":
                nnet = NNetWrapper(g, args)
                nnet.load_checkpoint(args.checkpoint, args.ckpt_file)
                mcts = MCTS(g, nnet, dotdict({"numMCTSSims": 800, "cpuct": 1.0}))
                return lambda x: np.argmax(mcts.getActionProb(x, temp=0))
            else:
                raise ValueError("not support player name {}".format(name))

        player1 = getPlayFunc(args.player1)
        player2 = getPlayFunc(args.player2)
        
        arena = game.Arena(player1, player2, g, display=g.display)
        results = arena.playGames(args.round, verbose=args.verbose)
        print("Final results: Player1 wins {}, Player2 wins {}, Draws {}".format(*results))


if __name__ == "__main__":
    main()