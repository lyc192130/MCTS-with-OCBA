"""


The board is represented by 
1: Player 2 
-1: Player 1
None: empty
For example, board (* means empty)
X, *, *
*, O, *
*, *, *

is represented by state = (-1, None, None, None, 1, None, None, None, None)
"""
from collections import namedtuple, defaultdict
from random import choice
from monte_carlo_tree_search_tic_tac_toe import MCTS, Node
from numpy import sqrt
import os
import argparse
import dill  # pip install dill

_Tree = namedtuple("Tree", "state terminal turn winner space")


class Tree(_Tree, Node):
    def find_children(tree):
        if tree.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in the next empty spots
        else:
            return {
                tree.make_move(i) for i in range(9) if tree.state[i] is None
            }

    def find_random_child(tree):
        possible_move = {i for i in range(9) if tree.state[i] is None}
        return tree.make_move(choice(tuple(possible_move)))

    def reward(tree, randomness=None):
        assert tree.terminal, "It's not a terminal node, check code!"

        if tree.winner == 1:
            return 1
        elif tree.winner == 0:
            return 0.5
        else:
            return 0

    def is_terminal(tree):
        return tree.terminal

    def make_move(tree, k):

        state = tree.state[:k] + (tree.turn,) + tree.state[k+1:]

        turn = -tree.turn
        winner = find_winner(state)
        space = tree.space-1
        is_terminal = (winner != 0) or (space == 0)
        if is_terminal and not (winner != 0 or all(s is not None for s in state)):
            print("case 1", Tree(state=state, terminal=is_terminal,
                                 turn=turn, winner=winner, space=space))
        if (winner != 0 or all(s is not None for s in state)) and not is_terminal:
            print("case 2", Tree(state=state, terminal=is_terminal,
                                 turn=turn, winner=winner, space=space))
        return Tree(state=state, terminal=is_terminal,
                    turn=turn, winner=winner, space=space)


def find_winner(state):
    winning_combos = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # cols
        [0, 4, 8], [2, 4, 6]  # diags
    ]
    for combo in winning_combos:
        s = 0
        for i in combo:
            if state[i] is None:
                break
            else:
                s += state[i]
        if s == 3 or s == -3:
            return 1 if s == 3 else -1
    return 0


def play_game_uct(budget=1000, exploration_weight=1, optimum=4, n0=5, opp='random', sigma_0=1, opp_first_move=0):
    mcts = MCTS(policy='uct', exploration_weight=exploration_weight,
                budget=budget, n0=n0, opp_policy=opp, sigma_0=sigma_0)
    tree = new_tree(budget=budget, opp_first_move=opp_first_move)

    for _ in range(budget):
        mcts.do_rollout(tree)

    next_tree = mcts.choose(tree)

    return (mcts, tree, next_tree)


def play_game_ocba(budget=1000, optimum=0, n0=5, opp='random', sigma_0=1, opp_first_move=0):
    mcts = MCTS(policy='ocba', budget=budget, n0=n0,
                opp_policy=opp, sigma_0=sigma_0)
    tree = new_tree(budget=budget, opp_first_move=opp_first_move)

    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)

    return (mcts, tree, next_tree)


def new_tree(budget=1000, opp_first_move=0):
    root = (None,)*opp_first_move + (-1,) + (None,)*(8-opp_first_move)
    return Tree(state=root, terminal=False, turn=1, winner=0, space=8)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("ckpt", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--rep', type=int,
                        help='number of replications', default=5000)
    parser.add_argument('--budget_start', type=int,
                        help='budget (number of rollouts) starts from (inclusive)', default=300)
    parser.add_argument('--budget_end', type=int,
                        help='budget (number of rollouts) end at (inclusive)', default=300)
    parser.add_argument('--step', type=int,
                        help='stepsize in experiment', default=50)
    parser.add_argument(
        '--n0', type=int, help='initial samples to each action', default=2)
    parser.add_argument('--sigma_0', type=int,
                        help='initial variance', default=10)
    parser.add_argument('--opp_policy', type=str,
                        help='opponent (Player 1) policy, must be either uct or random', default='uct')
    parser.add_argument('--opp_first_move', type=int,
                        help='the first move of opponent (Player 1), must be either 0 or 4 (correspond to setup 1 and 2, resp.)', default=0)
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
    opp_policy = args.opp_policy
    opp_first_move = args.opp_first_move
    results_uct = []
    results_ocba = []
    uct_selection = []
    ocba_selection = []
    exploration_weight = 1
    uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list = [], []
    uct_ave_Q_list, ocba_ave_Q_list = [], []
    uct_ave_std_list, ocba_ave_std_list = [], []
    uct_ave_std_corrected_list, ocba_ave_std_corrected_list = [], []
    if opp_first_move == 0:
        optimal_set = {4}
        setup = 1
    elif opp_first_move == 4:
        optimal_set = {0, 2, 6, 8}
        setup = 2
    else:
        raise ValueError('opp_first_move should either be 0 or 4.')

    ckpt = args.checkpoint

    if ckpt != '':
        dill.load_session(ckpt)
        # resume experiment from the last finished budget
        budget_range = range(budget+step, budget_end+1, step)

    for budget in budget_range:
        PCS_uct = 0
        PCS_ocba = 0
        uct_selection.append([])
        ocba_selection.append([])
        uct_visit_cnt, ocba_visit_cnt = defaultdict(int), defaultdict(int)
        uct_ave_Q, ocba_ave_Q = defaultdict(int), defaultdict(int)
        uct_ave_std, ocba_ave_std = defaultdict(int), defaultdict(int)
        for i in range(rep):
            uct_mcts, uct_root_node, uct_cur_node = play_game_uct(
                budget=budget,
                exploration_weight=exploration_weight,
                n0=n0,
                opp=opp_policy,
                sigma_0=sigma_0,
                opp_first_move=opp_first_move
            )
            PCS_uct += any(uct_cur_node.state[i] for i in optimal_set)

            ocba_mcts, ocba_root_node, ocba_cur_node = play_game_ocba(
                budget=budget,
                n0=n0,
                opp=opp_policy,
                sigma_0=sigma_0,
                opp_first_move=opp_first_move
            )
            PCS_ocba += any(ocba_cur_node.state[i] for i in optimal_set)
            '''
            Update the ave dict
            '''
            uct_visit_cnt.update(dict(
                (c, uct_visit_cnt[c]+uct_mcts.N[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_visit_cnt.update(dict(
                (c, ocba_visit_cnt[c]+ocba_mcts.N[c]) for c in ocba_mcts.children[ocba_root_node]))

            uct_ave_Q.update(dict(
                (c, uct_ave_Q[c]+uct_mcts.ave_Q[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_Q.update(dict(
                (c, ocba_ave_Q[c]+ocba_mcts.ave_Q[c]) for c in ocba_mcts.children[ocba_root_node]))

            uct_ave_std.update(dict((c, uct_ave_std_corrected[c]+sqrt(
                uct_mcts.std[c]**2 - sigma_0**2 / uct_mcts.N[c])) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std.update(dict((c, ocba_ave_std_corrected[c]+sqrt(
                ocba_mcts.std[c]**2 - sigma_0**2 / ocba_mcts.N[c])) for c in ocba_mcts.children[ocba_root_node]))

            if (i+1) % 100 == 0:
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
        ckpt_output = 'ckpt/tic_tac_toe_{opp_policy}_opponent_setup{setup}.pkl'.format(
            opp_policy=opp_policy, setup=setup)
        # dill.dump_session(ckpt_output)
        print('checkpoint saved!')
