"""
An example implementation of the abstract Node class for use in MCTS
for the inventory control problem
"""
from copy import deepcopy
from numpy.random import normal, binomial, randint
from collections import namedtuple, defaultdict
from random import choice, randrange
from monte_carlo_tree_search import MCTS, Node
from numpy import sqrt, savez
import matplotlib
import matplotlib.pyplot as plt

_TTTB = namedtuple("TicTacToeBoard", "state turn winner terminal")

_Tree = namedtuple("Tree", "state terminal num_actions stage action node_type cap horizons")
# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class Tree(_Tree, Node):
    def find_children(tree):
        if tree.terminal:  # If the game is finished then no moves can be made
            return set()
        # Otherwise, you can make a move in the next empty spots
        elif tree.node_type == 's':
            return {
                tree.make_move(i) for i in range(tree.num_actions+1) if tree.state + i <= tree.cap
            }
        else:
            raise "NOT a state node!"

    def find_random_child(tree):
        if tree.terminal:
            return None  # If the game is finished then no moves can be made
        if tree.node_type == 's':
            return tree.make_move( randint(1+tree.cap - tree.state))
        else:
            return tree.sample_next_state()

    def reward(tree, randomness = None):
        D = randomness
        cost = h*max([0, tree.state + tree.action - D]) \
                + p* max([0, D-tree.state-tree.action]) + K * (tree.action>0)
        return -cost

    def is_terminal(tree):
        return tree.terminal
        
    def make_move(tree, k):
        assert tree.node_type == 's', "The current node is not a state node!"
        
        state = tree.state
        action = k
        
        stage = tree.stage
        is_terminal = (stage == tree.horizons)
        return Tree(state=state, stage=stage, terminal=is_terminal, num_actions=tree.num_actions,\
                    action=action,  node_type='a', cap=20, horizons=tree.horizons)

    
    def sample_next_state(tree, path_reward = {}):
        assert tree.node_type == 'a', "The current node is not an action node!"
        D = randint(10)
        path_reward[tree] = tree.reward(D)
        stage = tree.stage+1
        action = tree.action
        state = max([tree.state + tree.action - D, 0])  
        
        is_terminal = (stage == tree.horizons)
        return Tree(state=state, stage=stage, terminal=is_terminal, num_actions=tree.num_actions,\
                    action=None, node_type='s', cap=20, horizons=tree.horizons)
        


def play_game_uct(budget=1000, exploration_weight = 1, optimum=0, n0=5, sigma_0=1):
    mcts = MCTS(policy='uct', exploration_weight=exploration_weight, budget=budget, n0=n0, sigma_0=sigma_0)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
        
    next_tree = mcts.choose(tree)
        
    return (mcts, tree, next_tree)


def play_game_ocba(budget=1000, optimum=0, n0=5, sigma_0=1):
    mcts = MCTS(policy='ocba',budget=budget, optimum=optimum, n0=n0, sigma_0=sigma_0)
    tree = new_tree()
    
    for _ in range(budget):
        mcts.do_rollout(tree)
    next_tree = mcts.choose(tree)
    
    return (mcts, tree, next_tree)


def new_tree():
    horizons = 3
    root = 5
    return Tree(state=root, stage=0, num_actions=20, terminal=False, action = None, node_type='s', cap = 20, horizons=horizons)


def lineplot_pred(x_data, x_label, y1_data, y2_data, y_label, title):
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
    # ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same axis
    ax2 = ax1
    ax2.plot(x_data, y2_data, color = y2_color, linestyle='solid', marker='o', label = 'OCBA')
    ax1.set_ylim([0,1])
    # Display legend
    ax1.legend(loc = 'lower right')
    plt.savefig(title+'.eps', format='eps')

if __name__ == "__main__":
    rep = 1000
    bud = 170
    budget_start = 50
    budget_end = 170
    step = 10
    budget_range = range(budget_start, budget_end+1, step)
    results_uct = []
    results_ocba = []
    uct_selection = []
    ocba_selection = []
    exploration_weight = 1
    h = 1
    p = 1
    K = 5
    optimum_K = {5:0, 0:3}
    optimum = 0
    n0 = 2
    sigma_0 = 100
    uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list = [], []
    uct_ave_Q_list, ocba_ave_Q_list = [], []
    uct_ave_std_list, ocba_ave_std_list = [], []
    uct_ave_std_corrected_list, ocba_ave_std_corrected_list = [], []
    
    
    for budget in budget_range:
        PCS_uct = 0
        PCS_ocba = 0
        uct_visit_cnt, ocba_visit_cnt = defaultdict(int), defaultdict(int)
        uct_ave_Q, ocba_ave_Q = defaultdict(int), defaultdict(int)
        uct_ave_std, ocba_ave_std = defaultdict(int), defaultdict(int)
        uct_ave_std_corrected, ocba_ave_std_corrected = defaultdict(int), defaultdict(int)
        for i in range(rep):
            uct_mcts, uct_root_node, uct_cur_node =\
                play_game_uct(budget=budget, exploration_weight=exploration_weight, optimum=optimum, n0=n0, sigma_0=sigma_0)
            PCS_uct += (uct_cur_node.action == optimum )
            
            ocba_mcts, ocba_root_node, ocba_cur_node = play_game_ocba(budget, n0=n0, optimum=optimum, sigma_0=sigma_0)
            PCS_ocba += (ocba_cur_node.action == optimum )
            '''
            Update the cnt dict
            '''
            uct_visit_cnt.update(dict((c, uct_visit_cnt[c]+uct_mcts.N[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_visit_cnt.update(dict((c, ocba_visit_cnt[c]+ocba_mcts.N[c]) for c in ocba_mcts.children[ocba_root_node]))
            
            uct_ave_Q.update(dict((c, uct_ave_Q[c]+uct_mcts.ave_Q[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_Q.update(dict((c, ocba_ave_Q[c]+ocba_mcts.ave_Q[c]) for c in ocba_mcts.children[ocba_root_node]))
            
            uct_ave_std.update(dict((c, uct_ave_std[c]+uct_mcts.std[c]) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std.update(dict((c, ocba_ave_std[c]+ocba_mcts.std[c]) for c in ocba_mcts.children[ocba_root_node]))
            
            uct_ave_std_corrected.update(dict((c, uct_ave_std_corrected[c]+sqrt(uct_mcts.std[c]**2 - sigma_0**2 / uct_mcts.N[c] )) for c in uct_mcts.children[uct_root_node]))
            ocba_ave_std_corrected.update(dict((c, ocba_ave_std_corrected[c]+sqrt(ocba_mcts.std[c]**2 - sigma_0**2 / ocba_mcts.N[c] )) for c in ocba_mcts.children[ocba_root_node]))
            if i%20 == 0:
                print('%0.2f%% finished for budget limit %d' %(100*i/rep, budget))
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
        '''
        cnt_ocba, cnt_uct = [0 for _ in range(20)], [0 for _ in range(20)]
        for u, o in zip(uct_selection[-1], uct_selection[-1]):
            cnt_ocba[int(o)] +=1
            cnt_uct[int(u)] +=1
        '''
    
    lineplot_pred(budget_range, 'N', results_uct, results_ocba,\
                    'PCS', 'results/Inventory_K{}_p{}'.format(K,p))
        
    savez('results/Inventory_K{}_p{}.npz'.format(K,p), budget_range = budget_range, rep=rep, results_uct=results_uct,\
        results_ocba=results_ocba, uct_visit_ave_cnt_list=uct_visit_ave_cnt_list, ocba_visit_ave_cnt_list=ocba_visit_ave_cnt_list,\
    uct_ave_Q_list=uct_ave_Q_list, ocba_ave_Q_list=ocba_ave_Q_list, uct_ave_std_list=uct_ave_std_list,\
        ocba_ave_std_list=ocba_ave_std_list, uct_ave_std_corrected_list=uct_ave_std_corrected_list, ocba_ave_std_corrected_list=ocba_ave_std_corrected_list,\
            sigma_0=sigma_0)
    
