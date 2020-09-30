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
from numpy.random import normal, binomial, randint, uniform, choice
from collections import namedtuple
from random import randrange
from monte_carlo_tree_search import MCTS, Node
from numpy import sqrt, savez, histogram
import matplotlib
import matplotlib.pyplot as plt

_TTTB = namedtuple("TicTacToeBoard", "state turn winner terminal")

_Tree = namedtuple("Tree", "state terminal num_actions stage action node_type")
# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class Tree(_Tree, Node):
    def find_children(tree):
        if tree.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in the next empty spots
        elif tree.node_type == 's':
            return {
                tree.make_move(i) for i in range(tree.num_actions)
            }
        else:
            raise "NOT a state node!"

    def find_random_child(tree):
        if tree.terminal:
            return None  # If the game is finished then no moves can be made
        if tree.node_type == 's':
            return tree.make_move( randrange(tree.num_actions))
        else:
            return tree.sample_next_state()

    def reward(tree, randomness = None):
        if tree.node_type == 's':
            return 0
        mean = 0
        if tree.action[-1] == None:
#            mean = tree.action[tree.stage]
#            sigma = sqrt(mean)
#            return binomial(n=1, p=tree.action[tree.stage]/100)
            return 0
        else:
            # for i in range(tree.stage):
            #     mean = mean + tree.state[i]
            mean = sum(tree.state[:-1])
#            mean = tree.action[-1]
#            sigma = sqrt(abs(mean))
            sigma = sqrt(abs(mean))
        
#        return normal(-mean, sigma+5)
#        return 1*binomial(n=1, p=1/(10*abs(mean)+1))
#        return 1*binomial(n=1, p=1/(tree.action[0]+1.1))
#        return normal(10*mean, 0)
#        return normal(mean, sigma)
#        return normal(0 if randomness==0 else -1, 0 if randomness==0 else 9)
        '''
        Chen et al 2006
        Example 2
        '''
#        if randomness == 0:
#            mean, sigma = 0, 3
#        elif randomness in (1,2):
#            mean = -0.4
#            sigma = (randomness*1.5)
#        elif randomness == 3:
#            mean, sigma = -1, 3
#        else:
#            mean, sigma = -2, 3
#        return normal(mean, sigma)
        '''
        Chen et al 2006
        Example 3
        '''
#        return normal(-randomness, 6)
        '''
        Uniform
        '''
#        return uniform(-10-randomness, 10-randomness)
        '''
        Made up tree
        '''
        return normal(-mean, 3)

    def is_terminal(tree):
        return tree.terminal
    
    def make_move(tree, k):
        assert tree.node_type == 's', "The current node is not a state node!"
        
        state = tree.state
        action = tree.action[:tree.stage] \
        + (k ,) \
        + tree.action[tree.stage+1:]
        
        stage = tree.stage
        is_terminal = stage == len(tree.state)
        return Tree(state=state, stage=stage, terminal=is_terminal, num_actions=tree.num_actions, action=action,  node_type='a')

    
    def sample_next_state(tree, path_reward = {}):
        assert tree.node_type == 'a', "The current node is not an action node!"
#        next_state = randint(3)+tree.action[tree.stage]
#        next_state = tree.action[tree.stage]
        
        '''
        w.p. 0.9, the next state = action (depite the current state)
        transit into other states with equal prob
        '''
        state_choices = [i for i in range(tree.num_actions)]
        p = 0.5
        prob_list = [(p if i == tree.action[tree.stage] else (1-p)/(tree.num_actions-1)) for i in range(tree.num_actions)]
        next_state = choice(state_choices, p=prob_list)
        
        randomness = next_state
        path_reward[tree] = tree.reward(randomness)
        stage = tree.stage+1
        action = tree.action
        state = tree.state[:stage] \
        + (next_state ,) \
        + tree.state[stage+1:]
        
        is_terminal = (stage+1 == len(tree.state))
        return Tree(state=state, stage=stage, terminal=is_terminal, num_actions=tree.num_actions, action=action, node_type='s')
        


def play_game_uct(budget=1000, exploration_weight = 1, n0=5, optimum=0):
    mcts = MCTS(policy='uct', exploration_weight=exploration_weight, budget=budget, n0=n0)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    '''
    Print out the results of the first layer
    '''
    def act(n):
        return n.action[n.stage]
    print('allocation under '+ mcts.policy,  ' # visited nodes = {}'.format(len(mcts.N)))
    if next_tree.action[0] != optimum:
        print("Wrong Selection!")
        # for n in sorted(mcts.children[tree], key=act):
        #     print(n, 'visits = {}, ave_Q = {}, std = {}'.format( mcts.N[n], mcts.ave_Q[n],mcts.std[n]))
        # print('\n\n')
    return next_tree.action[0]


def play_game_ocba(budget=1000, n0=5, optimum=0):
    mcts = MCTS(policy='ocba',budget=budget, n0=n0)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    '''
    Print out the results of the first layer
    '''
    def act(n):
        return n.action[n.stage]
    print('allocation under '+ mcts.policy,  ' # visited nodes = {}'.format(len(mcts.N)))
    if next_tree.action[0] != optimum:
        print("Wrong Selection!")
        # for n in sorted(mcts.children[tree], key=act):
        #     print(n, 'visits = {}, ave_Q = {}, std = {}'.format( mcts.N[n], mcts.ave_Q[n],mcts.std[n]))
        # print('\n\n')
    return next_tree.action[0]


def new_tree(budget=1000):
    horizons = 5
    root = (0,) + (None,)*horizons
    return Tree(state=root, stage=0, num_actions=3, terminal=False, action = (None,) * horizons, node_type='s')

def lineplot_pred(x_data, x_label, y1_data, y2_data, y_label, title):
    # Each variable will actually have its own plot object but they
    # will be displayed in just one plot
    # Create the first plot object and draw the line
    y1_color = '#539caf'
    y2_color = 'red'
#    p1 = y1_data/repetitions
#    p2 = y2_data/repetitions
#    error1 = sqrt(p1*(1-p1)/repetitions)
#    error2 = sqrt(p2*(1-p2)/repetitions)
    _, ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color = y1_color, label = "UCT")
    # Label axes
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same axis
    ax2 = ax1
    ax2.plot(x_data, y2_data, color = y2_color, label = 'OCBA')
    ax1.set_ylim([0,1])
    # Display legend
    ax1.legend(loc = 'upper right')

if __name__ == "__main__":
    rep = 50
    bud = 1500
    budget_start = 1000
    budget_end = 1000
    step = 500
    budget_range = range(budget_start, budget_end+1, step)
    results_uct = []
    results_ocba = []
    uct_selection = []
    ocba_selection = []
    exploration_weight = 1
    optimum = 0
    n0 = 10
    
    for budget in budget_range:
        PCS_uct = 0
        PCS_ocba = 0
        uct_selection.append([])
        ocba_selection.append([])
        for i in range(rep):
            x = play_game_uct(budget=budget, exploration_weight=exploration_weight, n0=n0, optimum=optimum)
            PCS_uct += 1 if x == optimum else 0
            uct_selection[-1].append(x)
            
            y = play_game_ocba(budget, n0=n0, optimum=optimum)
            PCS_ocba += 1 if y == optimum else 0
            ocba_selection[-1].append(y)
            if i%10 == 0:
                print('%0.2f%% finished' %(100*i/rep))
        print('PCS_uct = %0.3f, PCS_ocba = %0.3f' %(PCS_uct/rep, PCS_ocba/rep))
        results_uct.append(PCS_uct/rep)
        results_ocba.append(PCS_ocba/rep)
    
    lineplot_pred(budget_range, 'Number of samples', results_uct, results_ocba,\
                  'Error selection percentage', 'Selection errors')
#    savez('made_up_tree_results.npz', budget_range = budget_range, rep=rep, results_uct=results_uct,\
#             results_ocba=results_ocba)
    