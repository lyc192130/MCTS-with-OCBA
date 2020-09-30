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
from random import choice, randrange
from monte_carlo_tree_search_tic_tac_toe import MCTS, Node
from numpy import sqrt, savez
import matplotlib
import matplotlib.pyplot as plt
import shelve
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


_Tree = namedtuple("Tree", "state terminal turn winner")
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
        return tree.make_move(choice(list(possible_move)))

    def reward(tree, randomness = None):
        assert tree.terminal, "It's not a terminal node, check code!"
        
        return 1 if tree.winner == 1 else 0

    def is_terminal(tree):
        return tree.terminal
    
    def make_move(tree, k):
        
        state = tree.state[:k] + (tree.turn,) + tree.state[k+1:]
        
        turn = -tree.turn
        winner = find_winner(state)
        is_terminal = winner != 0 or all(s is not None for s in state)
        
        return Tree(state=state, terminal=is_terminal, turn=turn, winner=winner)

    
        
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
        


def play_game_uct(budget=1000, exploration_weight = 1, optimum=4, n0=5, opp='random'):
    mcts = MCTS(policy='uct', exploration_weight=exploration_weight, budget=budget, n0=n0, opp_policy=opp)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
        
    next_tree = mcts.choose(tree)
    
    return (mcts, tree, next_tree)


def play_game_ocba(budget=1000, optimum=0, n0=5, opp='random'):
    mcts = MCTS(policy='ocba',budget=budget, optimum=optimum, n0=n0, opp_policy=opp)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    
    
    
    return (mcts, tree, next_tree)


def new_tree(budget=1000):
    root = (-1,)+(None,)*8
    return Tree(state=root, terminal=False, turn=1, winner=0)

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
    ax1.set_ylim([0.7,1])
    # Display legend
    ax1.legend(loc = 'lower right')
    plt.savefig('results/tic_tac_toe_{}_opponent.eps'.format(opp_policy), format='eps')

def allocation_dist_plot(actions, ave_Q, ave_std, ave_N, title):
    '''

    Parameters
    ----------
    actions : list of ints
        the children actions.
    ave_Q : list of floats
    ave_std : list of floats
    ave_N : list of floats

    Returns
    -------
    None. Plot and save the distribution of ave_N with other variables on the same plot.
    
    Reference: https://matplotlib.org/examples/axes_grid/demo_parasite_axes2.html
    choose color from https://matplotlib.org/tutorials/colors/colors.html
    '''
    
    # host for ave_Q, par1 for ave_std, par2 for ave_N
    host = host_subplot(111, axes_class=AA.Axes)
    # plt.subplots_adjust(right=0.75)
    
    par1 = host.twinx()
    par2 = host.twinx()
    
    offset = 50
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par1.axis['right'] = par1.get_grid_helper().new_fixed_axis(loc='right', axes=par1, offset=(0, 0))
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))
    
    par2.axis["right"].toggle(all=True)
    
    host.set_xlim(actions[0]-0.5, actions[-1]+0.5)
    host.set_ylim(0.9*min(ave_Q), 1.3*max(ave_Q))
    par1.set_ylim(0, 1.1*max(ave_std))
    par2.set_ylim(0, 1.1*max(ave_N))
    
    host.set_xlabel("Actions")
    host.set_ylabel("Estimated value function Q")
    par1.set_ylabel("Estimated standard deviation")
    par2.set_ylabel("Average number of visits")
    
    p1, = host.plot(actions, ave_Q, label="Q", linestyle='solid', marker='o')
    p2, = par1.plot(actions, ave_std, label="std", linestyle='dashed', marker='v')
    p3 = par2.bar(actions, ave_N, label="# visits", color='cyan')
    
    
    
    host.legend(loc=(0.7, 0.5))
    
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.patches[0].get_facecolor())
    
    fig = plt.gcf()
    plt.draw()
    plt.show()
    fig.savefig(title, format='eps', bbox_inches='tight')
    
    
    
uct_ave_Q_to_list, uct_ave_std_to_list, uct_ave_N_to_list = [], [], []
ocba_ave_Q_to_list, ocba_ave_std_to_list, ocba_ave_N_to_list = [], [], []
actions = range(1,9)
def sort_key(n):
    return sum([i for i in range(9) if n.state[i]])
for c in sorted(uct_mcts.children[uct_root_node], key=sort_key):
    uct_ave_Q_to_list.append(uct_ave_Q[c])
    uct_ave_std_to_list.append(uct_ave_std_corrected[c])
    uct_ave_N_to_list.append(uct_visit_cnt[c])
    print("UCT action = %d: ave_Q = %0.2f, ave_std = %0.2f, ave_N = %0.2f" \
          %(sort_key(c), uct_ave_Q[c], uct_ave_std[c], uct_visit_cnt[c]))
        
        
for c in sorted(uct_mcts.children[uct_root_node], key=sort_key):
    ocba_ave_Q_to_list.append(ocba_ave_Q[c])
    ocba_ave_std_to_list.append(ocba_ave_std_corrected[c])
    ocba_ave_N_to_list.append(ocba_visit_cnt[c])
    print("OCBA action = %d: ave_Q = %0.2f, ave_std = %0.2f, ave_N = %0.2f" \
          %(sort_key(c), ocba_ave_Q[c], ocba_ave_std[c], ocba_visit_cnt[c]))
        

allocation_dist_plot(
    actions=actions,
    ave_Q=uct_ave_Q_to_list,
    ave_std=uct_ave_std_to_list,
    ave_N=uct_ave_N_to_list,
    title='results/TTT_sample_distribution_{}_opponent_uct2.eps'.format(opp_policy)
    )
allocation_dist_plot(
    actions=actions,
    ave_Q=ocba_ave_Q_to_list,
    ave_std=ocba_ave_std_to_list,
    ave_N=ocba_ave_N_to_list,
    title='results/TTT_sample_distribution_{}_opponent_ocba2.eps'.format(opp_policy)
    )

lineplot_pred(budget_range, 'N', results_uct, results_ocba,\
                'PCS', '', opp_policy=opp_policy)

for uct_ave_N in uct_visit_ave_cnt_list:
    print(sum(uct_ave_N.values()))



    