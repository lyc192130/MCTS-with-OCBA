"""
A minimal implementation of Monte Carlo tree search (MCTS) in Python 3
Luke Harold Miles, July 2019, Public Domain Dedication
See also https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
import math
import bottleneck as bn
from numpy import nanstd, nanmean, mean, argmin, ceil, ones, array, log, std,\
 nanmin, nanmax, nanargmin, argmax, sqrt, floor, isnan
import operator
from random import choice, sample
import numpy as np



class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=5000, policy='uct', budget=1000, optimum=0, n0=5, opp_policy='random', sigma_0=1):
        self.Q = defaultdict(int)  # total reward of each node
        self.V_bar = defaultdict(int)
        self.V_hat = defaultdict(int)
        self.N = defaultdict(int)  # total visit count for each node
        self.children = defaultdict(set)  # children of each node
        self.exploration_weight = exploration_weight
        self.all_Q = defaultdict(list)
        assert policy in {'uct', 'ocba'}, 'Policy must be either uct or ocba!'
        self.policy = policy
        self.std = defaultdict(int)  # std of each node
        self.ave_Q = defaultdict(int)
        self.budget=budget
        self.initial_n0 = n0
        self.n0 = n0
        self.leaf_cnt = defaultdict(int)
        self.opp_policy = opp_policy
        self.sigma_0 = sigma_0
        

    def choose(self, node):
        "Choose the best successor of node. (Choose a move in the game)"
        if node.is_terminal():
            raise RuntimeError(f"choose called on terminal node {node}")

        if node not in self.children:
            return node.find_random_child()

        def score(n):
            if self.N[n] == 0:
                return float("-inf")  # avoid unseen moves
            return self.ave_Q[n]  # average reward
        rtn = max(self.children[node], key=score)
        
        return rtn

    def do_rollout(self, node):
        "Make the tree one layer better. (Train for one iteration.)"
        
        path = self._select(node)
        leaf = path[-1]
        
        self.leaf_cnt[leaf] += 1
        sim_reward = self._simulate(leaf)
        self._backpropagate(path, sim_reward)

    def _select(self, node):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            self.N[node] += 1
            if node.terminal:
                # node is either unexplored or terminal
                return path
            
            if len(self.children[node]) == 0:
                # First time to visit a state node, add its children to self.children
                children_found = node.find_children()
                self.children[node] = children_found
                
            
            if node.turn == -1:
                # opponent's turn
                if self.opp_policy == 'random':
                    '''
                    Random strategy
                    '''
                    node = node.find_random_child()
                elif self.opp_policy == 'uct':
                    '''
                    UCT
                    '''
                    
                    expandable = [n for n in self.children[node] if self.N[n] < 1]
                    if expandable:
                        node = expandable.pop()
                    else:
                        log_N_vertex = math.log(sum([self.N[c] for c in self.children[node]]))
                        node = min(self.children[node], key=lambda n:self.ave_Q[n] 
                                   - self.exploration_weight * math.sqrt( 2 * log_N_vertex / self.N[n]))
                continue
            
                
            
            # self.n0 = max([2, self.initial_n0 - 2*node.space])
            expandable = [n for n in self.children[node] if self.N[n] < self.n0]
            
            if  expandable:
                # expandable
                a = self._expand(node)
                if len(self.children[a]) == 0:
                    self.children[a] = a.find_children()
                path.append(a)
                self.N[a] += 1
                
                return path
            else:
                if len(self.children[node]) == 0:
                    print("empty")
                if self.policy == 'uct':
                    a = self._uct_select(node)  # descend a layer deeper
                else:
                    a = self._ocba_select(node)
                # path.append(a)
                node = a

    def _expand(self, node, path_reward=None):
        "Add a node to the tree"
        explored_once = [n for n in self.children[node] if self.N[n] < self.n0]
        return explored_once.pop()

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        while True:
            if not node.is_terminal():
                node = node.find_random_child()
            if node.terminal:
                return node.reward()

    def _backpropagate(self, path, r):
        "Send the reward back up to the ancestors of the leaf"
        for i in range(len(path)-1, -1, -1):
            node = path[i]
            if i != len(path)-1 and ( path[i].terminal):
                print(path)
                assert not path[i].terminal, "Intermediate node should not be terminal!"
            '''
            Iteratively update std, which is supposed to be faster.
            Population std.
            '''
            self.Q[node] += r
            self.all_Q[node].append(r) # This is not necessary anymore with iterative update.
            old_ave_Q = self.ave_Q[node]
            self.ave_Q[node] = self.Q[node] / self.N[node]
            self.std[node] = self.sigma_0 if self.N[node] == 1 else sqrt(((self.N[node]-1)*self.std[node]**2 + (r - old_ave_Q) * (r - self.ave_Q[node]))/self.N[node])
            
            

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
#        if not all(n in self.children for n in self.children[node]):
#            print(node)
        assert all(n in self.children for n in self.children[node])

        log_N_vertex = math.log(sum([self.N[c] for c in self.children[node]]))

        def uct(n):
            "Upper confidence bound for trees"
            return self.ave_Q[n] + self.exploration_weight * math.sqrt(
                2 * log_N_vertex / self.N[n]
            )

        return max(self.children[node], key=uct)
    
    def _ocba_select(self, node):
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node])>0, "Error! Empty children action set!"
        
        if len(self.children[node]) == 1:
            return list(self.children[node])[0]
        b = max(self.children[node], key=lambda n:self.ave_Q[n])
        best_actions_set = {n for n in self.children[node] if self.ave_Q[n] == self.ave_Q[b]}
        delta = defaultdict(int)
        delta.update(dict( (k, abs(self.ave_Q[k] - self.ave_Q[b])) for k in self.children[node] ))

        
        '''
        Method II to find para
        '''
        suboptimals_set = self.children[node] - best_actions_set
        if len(suboptimals_set) == 0:
            return min(self.children[node], key=lambda n: self.N[n])
        ref = choice(list(suboptimals_set))
        
        para_sum = 0
        para = defaultdict(int)
        para.update(dict( \
                          (k, ((self.std[k]/delta[k])/(self.std[ref]/delta[ref]))**2 \
                          ) \
                          for k in suboptimals_set \
                        )\
                    )
           
        para.update(dict( (k,  sqrt(\
                                    sum( \
                                        [(self.std[k]*para[c]/self.std[c])**2 for c in suboptimals_set]\
                                        ) \
                                    ) \
                            )\
                          for k in best_actions_set \
                        )\
                    )

        para_sum = sum(para.values())
        para[ref] = 1
       
       
        # totalBudget = self.N[node]
        totalBudget = sum([self.N[c] for c in self.children[node]])+1
        ref_sol = (totalBudget)/para_sum
        
        return max(self.children[node], key=lambda n:para[n]*ref_sol - self.N[n])
    

        
class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    @abstractmethod
    def find_children(self):
        "All possible successors of this board state"
        return set()

    @abstractmethod
    def find_random_child(self):
        "Random successor of this board state (for more efficient simulation)"
        return None

    @abstractmethod
    def is_terminal(self):
        "Returns True if the node has no children"
        return True

    @abstractmethod
    def reward(self):
        "Assumes `self` is terminal node. 1=win, 0=loss, .5=tie, etc"
        return 0

    @abstractmethod
    def __hash__(self):
        "Nodes must be hashable"
        return 123456789

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
    
    
    
            