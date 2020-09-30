"""
An example implementation of the abstract Node class for use in MCTS

If you run this file then you can play against the computer.

A tic-tac-toe board is represented as a tuple of 9 values, each either None,
True, or False, respectively meaning 'empty', 'X', and 'O'.

The board is indexed by row:
0 1 2
3 4 5
6 7 8

For example, this game board
O - X
O X -
X - -
corrresponds to this tuple:
(False, None, True, False, True, None, True, None, None)
"""
from copy import deepcopy
from numpy.random import normal, binomial, randint
from collections import namedtuple, defaultdict
from random import choice, randrange, sample
from monte_carlo_tree_search_tic_tac_toe import MCTS, Node
from numpy import sqrt, savez
import matplotlib
import matplotlib.pyplot as plt
import shelve


_Tree = namedtuple("Tree", "state terminal turn winner space")
# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
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

    def reward(tree, randomness = None):
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
                      [0,1,2], [3,4,5], [6,7,8], # rows
                      [0,3,6], [1,4,7], [2,5,8], # cols
                      [0,4,8], [2,4,6] # diags
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
        


def play_game_uct(budget=1000, exploration_weight = 1, optimum=4, n0=5, opp='random', sigma_0=1):
    mcts = MCTS(policy='uct', exploration_weight=exploration_weight, budget=budget, n0=n0, opp_policy=opp, sigma_0=sigma_0)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
        
    next_tree = mcts.choose(tree)
    
    return (mcts, tree, next_tree)


def play_game_ocba(budget=1000, optimum=0, n0=5, opp='random', sigma_0=1):
    mcts = MCTS(policy='ocba',budget=budget, optimum=optimum, n0=n0, opp_policy=opp, sigma_0=sigma_0)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    
    
    return (mcts, tree, next_tree)


def new_tree(budget=1000):
    root = (-1,)+(None,)*8
    return Tree(state=root, terminal=False, turn=1, winner=0, space=8)

def lineplot_pred(x_data, x_label, y1_data, y2_data, y_label, title, opp_policy):
    # Each variable will actually have its own plot object but they
    # will be displayed in just one plot
    # Create the first plot object and draw the line
    y1_color = '#539caf'
    y2_color = 'red'
    _, ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color = y1_color, linestyle='dashed', marker='v', label = "UCT")
    # Label axes
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same axis
    ax2 = ax1
    ax2.plot(x_data, y2_data, color = y2_color, linestyle='solid', marker='o', label = 'OCBA')
    ax1.set_ylim([0.6,1])
    # Display legend
    ax1.legend(loc = 'lower right')

if __name__ == "__main__":
    rep = 100
    bud = 100
    budget_start = 700
    budget_end = 700
    step = 50
    budget_range = range(budget_start, budget_end+1, step)
    results_uct = []
    results_ocba = []
    uct_selection = []
    ocba_selection = []
    exploration_weight = 1
    optimum = 4
    n0 = 2
    sigma_0 = 100
    uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list = [], []
    uct_ave_Q_list, ocba_ave_Q_list = [], []
    uct_ave_std_list, ocba_ave_std_list = [], []
    uct_ave_std_corrected_list, ocba_ave_std_corrected_list = [], []
    opp_policy = 'random'
    
    for budget in budget_range:
        PCS_uct = 0
        PCS_ocba = 0
        uct_selection.append([])
        ocba_selection.append([])
        uct_visit_cnt, ocba_visit_cnt = defaultdict(int), defaultdict(int)
        uct_ave_Q, ocba_ave_Q = defaultdict(int), defaultdict(int)
        uct_ave_std, ocba_ave_std = defaultdict(int), defaultdict(int)
        uct_ave_std_corrected, ocba_ave_std_corrected = defaultdict(int), defaultdict(int)
        for i in range(rep):
            uct_mcts, uct_root_node, uct_cur_node =\
                play_game_uct(budget=budget, exploration_weight=exploration_weight, optimum=optimum, n0=n0, opp=opp_policy, sigma_0=sigma_0)
            PCS_uct += (uct_cur_node.state[4] == 1 )
            # uct_selection[-1].append(uct_cur_node.action)
            
            ocba_mcts, ocba_root_node, ocba_cur_node = play_game_ocba(budget, n0=n0, optimum=optimum, opp=opp_policy, sigma_0=sigma_0)
            PCS_ocba += (ocba_cur_node.state[4] == 1 )
            # ocba_selection[-1].append(ocba_cur_node.action)
            '''
            Update the ave dict
            '''
            uct_visit_cnt.update(dict((c, uct_visit_cnt[c]+uct_mcts.N[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_visit_cnt.update(dict((c, ocba_visit_cnt[c]+ocba_mcts.N[c]) for c in ocba_mcts.children[ocba_root_node]))
            
            uct_ave_Q.update(dict((c, uct_ave_Q[c]+uct_mcts.ave_Q[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_Q.update(dict((c, ocba_ave_Q[c]+ocba_mcts.ave_Q[c]) for c in ocba_mcts.children[ocba_root_node]))
            
            uct_ave_std.update(dict((c, uct_ave_std[c]+uct_mcts.std[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std.update(dict((c, ocba_ave_std[c]+ocba_mcts.std[c]) for c in ocba_mcts.children[ocba_root_node]))
            
            uct_ave_std_corrected.update(dict((c, uct_ave_std_corrected[c]+sqrt(uct_mcts.std[c]**2 - sigma_0**2 / uct_mcts.N[c] )) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std_corrected.update(dict((c, ocba_ave_std_corrected[c]+sqrt(ocba_mcts.std[c]**2 - sigma_0**2 / ocba_mcts.N[c] )) for c in ocba_mcts.children[ocba_root_node]))
            # print([(ocba_mcts.std[c]**2 - sigma_0**2 / ocba_mcts.N[c] ) for c in ocba_mcts.children[ocba_root_node] if (ocba_mcts.std[c]**2 - sigma_0**2 / ocba_mcts.N[c] ) < 0])
            if i%100 == 0:
                print('%0.2f%% finished for budget limit %d' %(100*(i+1)/rep, budget))
                print('Current PCS: uct=%0.3f, ocba=%0.3f' %(PCS_uct/(i+1), (PCS_ocba/(i+1))))
        
        print("Budget %d has finished" %(budget))
        print('PCS_uct = %0.3f, PCS_ocba = %0.3f' %(PCS_uct/rep, PCS_ocba/rep))
        
        uct_visit_cnt.update(dict((c, uct_visit_cnt[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_visit_cnt.update(dict((c, ocba_visit_cnt[c]/rep) for c in ocba_mcts.children[ocba_root_node]))
        
        uct_ave_Q.update(dict((c, uct_ave_Q[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_ave_Q.update(dict((c, ocba_ave_Q[c]/rep) for c in ocba_mcts.children[ocba_root_node]))
        
        uct_ave_std.update(dict((c, uct_ave_std[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_ave_std.update(dict((c, ocba_ave_std[c]/rep) for c in ocba_mcts.children[ocba_root_node]))
        
        uct_ave_std_corrected.update(dict((c, uct_ave_std_corrected[c]/rep) for c in uct_mcts.children[uct_root_node]))
        ocba_ave_std_corrected.update(dict((c, ocba_ave_std_corrected[c]/rep) for c in ocba_mcts.children[ocba_root_node]))
        
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
        
    savez('results/tic_tac_toe_all_variables_{}_opponent.npz'.format(opp_policy), budget_range = budget_range, rep=rep, results_uct=results_uct,\
        results_ocba=results_ocba, uct_visit_ave_cnt_list=uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list=ocba_visit_ave_cnt_list,\
    uct_ave_Q_list=uct_ave_Q_list, ocba_ave_Q_list=ocba_ave_Q_list, uct_ave_std_list=uct_ave_std_list,\
        ocba_ave_std_list=ocba_ave_std_list)
    