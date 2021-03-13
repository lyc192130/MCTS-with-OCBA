"""
An example implementation of the abstract Node class for use in MCTS
for the inventory control problem
"""
from copy import deepcopy
from numpy.random import normal, binomial, randint, random
from collections import namedtuple, defaultdict
from random import choice, randrange
from monte_carlo_tree_search import MCTS, Node
from numpy import sqrt, savez
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import argparse
import dill # pip install dill


_Tree = namedtuple(
    "_Tree", "state terminal num_actions stage action node_type cap horizons")



class Tree(_Tree, Node):
    def find_children(self):
        # If the game is finished then no moves can be made
        if self.terminal:  
            return set()
        # Otherwise, you can make a move in the next empty spots
        elif self.node_type == 's':
            return {
                self.make_move(i) for i in range(self.num_actions+1) if self.state + i <= self.cap
            }
        else:
            raise "NOT a state node!"

    def find_random_child(self):
        if self.terminal:
            return None  # If the game is finished then no moves can be made
        if self.node_type == 's':
            # faster than np.random.randint(1+self.cap - self.state)
            return self.make_move(int((1+self.cap-self.state) * random() ))
        else:
            return self.sample_next_state()

    def reward(self, randomness=None):
        D = randomness
        cost = h*max([0, self.state + self.action - D]) \
            + p * max([0, D-self.state-self.action]) + K * (self.action > 0)
        return -cost

    def is_terminal(self):
        return self.terminal

    def make_move(self, k):
        assert self.node_type == 's', "The current node is not a state node!"

        state = self.state
        action = k

        stage = self.stage
        is_terminal = (stage == self.horizons)
        return Tree(state=state, stage=stage, terminal=is_terminal, num_actions=self.num_actions,
                    action=action,  node_type='a', cap=20, horizons=self.horizons)

    def sample_next_state(self, path_reward={}):
        assert self.node_type == 'a', "The current node is not an action node!"
        D = int(10*random())
        path_reward[self] = self.reward(D)
        stage = self.stage+1
        state = max(self.state + self.action - D, 0)

        is_terminal = (stage == self.horizons)
        return Tree(state=state, stage=stage, terminal=is_terminal, num_actions=self.num_actions,
                    action=None, node_type='s', cap=20, horizons=self.horizons)


def play_game_uct(budget=1000, exploration_weight=1, optimum=0, n0=2, sigma_0=1):
    mcts = MCTS(policy='uct', exploration_weight=exploration_weight,
                budget=budget, n0=n0, sigma_0=sigma_0)
    tree = new_tree()

    for _ in range(budget):
        mcts.do_rollout(tree)

    next_tree = mcts.choose(tree)

    return (mcts, tree, next_tree)


def play_game_ocba(budget=1000, optimum=0, n0=5, sigma_0=1):
    mcts = MCTS(policy='ocba', budget=budget,
                optimum=optimum, n0=n0, sigma_0=sigma_0)
    tree = new_tree()

    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)

    return (mcts, tree, next_tree)


def new_tree():
    horizons = 3
    root = 5
    return Tree(state=root, stage=0, num_actions=20, terminal=False, action=None, node_type='s', cap=20, horizons=horizons)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int,
                        help='number of replications', default=1500)
    parser.add_argument('--budget_start', type=int,
                        help='budget (number of rollouts) starts from (inclusive)', default=50)
    parser.add_argument('--budget_end', type=int,
                        help='budget (number of rollouts) end at (inclusive)', default=200)
    parser.add_argument('--step', type=int,
                        help='stepsize in experiment', default=10)
    parser.add_argument(
        '--n0', type=int, help='initial samples to each action', default=2)
    parser.add_argument('--sigma_0', type=int,
                        help='initial variance', default=100)
    parser.add_argument(
        '--p', type=int, help='invetory control penalty cost', default=1)
    parser.add_argument(
        '--K', type=int, help='invetory control fixed ordering cost', default=5)
    parser.add_argument(
        '--checkpoint', type=str, help='relative path to checkpoint', default='')
    
    
    args = parser.parse_args()
    
    rep = args.rep 
    budget_start = args.budget_start
    budget_end = args.budget_end
    step = args.step
    budget_range = range(budget_start, budget_end+1, step)
    n0 = args.n0
    sigma_0 = args.sigma_0
    p = args.p 
    K = args.K
    results_uct = []
    results_ocba = []
    uct_selection = []
    ocba_selection = []
    exploration_weight = 1
    h = 1
    if p == 1 and K == 5:
        optimum = 0
    elif p == 10 and K == 0:
        optimum = 4
    else:
        raise ValueError('p and K must be either (1, 5) or (10, 0)')
    uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list = [], []
    uct_ave_Q_list, ocba_ave_Q_list = [], []
    uct_ave_std_list, ocba_ave_std_list = [], []
    uct_ave_std_corrected_list, ocba_ave_std_corrected_list = [], []
    ckpt = args.checkpoint

    if ckpt != '':
        dill.load_session(ckpt)
        # start experiment from the last finished budget
        budget_range = range(budget+step, budget_end+1, step)
    for budget in budget_range:
        PCS_uct = 0
        PCS_ocba = 0
        uct_visit_cnt, ocba_visit_cnt = defaultdict(int), defaultdict(int)
        uct_ave_Q, ocba_ave_Q = defaultdict(int), defaultdict(int)
        uct_ave_std, ocba_ave_std = defaultdict(int), defaultdict(int)
        uct_ave_std_corrected, ocba_ave_std_corrected = defaultdict(
            int), defaultdict(int)
        for i in range(rep):
            uct_mcts, uct_root_node, uct_cur_node =\
                play_game_uct(budget=budget, exploration_weight=exploration_weight,
                              optimum=optimum, n0=n0, sigma_0=sigma_0)
            PCS_uct += (uct_cur_node.action == optimum)

            ocba_mcts, ocba_root_node, ocba_cur_node = play_game_ocba(
                budget, n0=n0, optimum=optimum, sigma_0=sigma_0)
            PCS_ocba += (ocba_cur_node.action == optimum)
                
            '''
            Update the cnt dict
            '''
            uct_visit_cnt.update(dict(
                (c, uct_visit_cnt[c]+uct_mcts.N[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_visit_cnt.update(dict(
                (c, ocba_visit_cnt[c]+ocba_mcts.N[c]) for c in ocba_mcts.children[ocba_root_node]))

            uct_ave_Q.update(dict(
                (c, uct_ave_Q[c]+uct_mcts.ave_Q[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_Q.update(dict(
                (c, ocba_ave_Q[c]+ocba_mcts.ave_Q[c]) for c in ocba_mcts.children[ocba_root_node]))

            uct_ave_std.update(dict(
                (c, uct_ave_std[c]+uct_mcts.std[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std.update(dict(
                (c, ocba_ave_std[c]+ocba_mcts.std[c]) for c in ocba_mcts.children[ocba_root_node]))

            uct_ave_std_corrected.update(dict((c, uct_ave_std_corrected[c]+sqrt(
                uct_mcts.std[c]**2 - sigma_0**2 / uct_mcts.N[c])) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std_corrected.update(dict((c, ocba_ave_std_corrected[c]+sqrt(
                ocba_mcts.std[c]**2 - sigma_0**2 / ocba_mcts.N[c])) for c in ocba_mcts.children[ocba_root_node]))
            if (i+1) % 20 == 0:
                print('%0.2f%% finished for budget limit %d' %
                      (100*(i+1)/rep, budget))
                print('Current PCS: uct=%0.3f, ocba=%0.3f' %
                      (PCS_uct/(i+1), (PCS_ocba/(i+1))))

        uct_visit_cnt.update(
            dict((c, uct_visit_cnt[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_visit_cnt.update(
            dict((c, ocba_visit_cnt[c]/rep) for c in ocba_mcts.children[ocba_root_node]))

        uct_ave_Q.update(dict((c, uct_ave_Q[c]/rep)
                              for c in uct_mcts.children[uct_root_node]))
        ocba_ave_Q.update(dict((c, ocba_ave_Q[c]/rep)
                               for c in ocba_mcts.children[ocba_root_node]))

        uct_ave_std.update(
            dict((c, uct_ave_std[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_ave_std.update(
            dict((c, ocba_ave_std[c]/rep) for c in ocba_mcts.children[ocba_root_node]))

        uct_ave_std_corrected.update(dict(
            (c, uct_ave_std_corrected[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_ave_std_corrected.update(dict(
            (c, ocba_ave_std_corrected[c]/rep) for c in ocba_mcts.children[ocba_root_node]))

        uct_visit_ave_cnt_list.append(uct_visit_cnt)
        ocba_visit_ave_cnt_list.append(ocba_visit_cnt)

        uct_ave_Q_list.append(uct_ave_Q)
        ocba_ave_Q_list.append(ocba_ave_Q)

        uct_ave_std_list.append(uct_ave_std)
        ocba_ave_std_list.append(ocba_ave_std)

        uct_ave_std_corrected_list.append(uct_ave_std_corrected)
        ocba_ave_std_corrected_list.append(ocba_ave_std_corrected)

        results_uct.append(PCS_uct/rep)
        results_ocba.append(PCS_ocba/rep)
    
        print("Budget %d has finished" % (budget))
        print('PCS_uct = %0.3f, PCS_ocba = %0.3f' %
              (PCS_uct/rep, PCS_ocba/rep))
        ckpt_output = 'ckpt/Inventory_K{}_p{}_budget_{}.pkl'.format(K, p, budget)
        dill.dump_session(ckpt_output)
        print('checkpoint saved!')

