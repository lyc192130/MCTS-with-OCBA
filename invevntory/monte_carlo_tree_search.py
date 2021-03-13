"""
Referenced
https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
"""
from abc import ABC, abstractmethod
from collections import defaultdict
from numpy import log, sqrt


class MCTS:
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(self, exploration_weight=5000, policy='uct', budget=1000, optimum=0, n0=5, sigma_0=1):
        self.Q = defaultdict(float)  # total reward of each node
        self.V_bar = defaultdict(float)
        self.V_hat = defaultdict(float)
        self.N = defaultdict(int)  # total visit count for each node
        self.children = defaultdict(set)  # children of each node
        self.exploration_weight = exploration_weight
        # self.all_Q = defaultdict(list)
        assert policy in {'uct', 'ocba'}, 'Policy must be either uct or ocba!'
        self.policy = policy
        self.std = defaultdict(float)  # std of each node
        self.ave_Q = defaultdict(float)
        self.budget = budget
        self.initial_n0 = n0
        self.n0 = n0
        self.leaf_cnt = defaultdict(int)
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
        path_reward = defaultdict(int)
        path = self._select(node, path_reward)
        leaf = path[-1]
        self.leaf_cnt[leaf] += 1
        '''
        TODO remove expand
        self._expand(leaf)
        '''
        sim_reward = self._simulate(leaf)
        self._backpropagate(path, sim_reward, path_reward)

    def _select(self, node, path_reward):
        "Find an unexplored descendent of `node`"
        path = []
        while True:
            path.append(node)
            self.N[node] += 1
            if node.terminal:
                # node is either unexplored or terminal
                return path

            if len(self.children[node]) == 0 and node.node_type == 's':
                # First time to visit a state node, add its children to self.children
                children_found = node.find_children()
                self.children[node] = children_found
                for c in children_found:
                    self.children[c].add(c.sample_next_state())

            self.n0 = self.initial_n0 if node.stage == 0 else 2
            expandable = [n for n in self.children[node]
                          if self.N[n] < self.n0]

            if expandable:
                # expandable
                a = self._expand(node)
                path.append(a)
                self.N[a] += 1
                y = self._expand(a, path_reward)
                self.children[a].add(y)
                path.append(y)
                self.N[y] += 1
                if len(self.children[y]) == 0 and y.node_type == 's':
                    # First time to visit a state node, add its children to self.children
                    children_found = y.find_children()
                    self.children[y] = children_found
                    for c in children_found:
                        self.children[c].add(c.sample_next_state())

                return path
            else:
                if self.policy == 'uct':
                    a = self._uct_select(node)  # descend a layer deeper
                else:
                    a = self._ocba_select(node)
                path.append(a)
                self.N[a] += 1
                node = self._expand(a, path_reward)
                self.children[a].add(node)

    def _expand(self, node, path_reward=None):
        "Add a node to the tree"
        if node.node_type == 's':
            under_explored = [n for n in self.children[node]
                              if self.N[n] < self.n0]
            return under_explored.pop()
        else:
            assert path_reward is not None, "Please include path_reward when expanding an action"
            return node.sample_next_state(path_reward)

    def _simulate(self, node):
        "Returns the reward for a random simulation (to completion) of `node`"
        reward = 0
        sim_reward_dict = defaultdict(int)
        while True:
            if not node.is_terminal():
                node = node.find_random_child()
                next_node = node.sample_next_state(sim_reward_dict)
                reward += sim_reward_dict[node]
                node = next_node
            if node.terminal:
                return reward

    def _backpropagate(self, path, sim_reward, path_reward):
        "Send the reward back up to the ancestors of the leaf"
        leaf = path[-1]
        self.leaf_cnt[leaf] += 1
        assert leaf.node_type == 's', "The leaf node should be a state node!"
        self.V_hat[leaf] = ((self.N[leaf]-1)/self.N[leaf]) * \
            self.V_hat[leaf] + sim_reward/self.N[leaf]
        for i in range(len(path)-2, -1, -1):
            node, child = path[i], path[i+1]
            if node.node_type == 's':
                # Current node is a state node
                alpha = 1 - 1/(5*self.N[node])
                self.V_bar[node] = ((self.N[node]-1)/self.N[node]) * \
                    self.V_bar[node] + self.ave_Q[child]/self.N[node]
                self.V_hat[node] = (1-alpha) * self.V_bar[node] + alpha * max(
                    [self.ave_Q[c] for c in self.children[node] if self.N[c] > 0])
            else:
                # Current node is an action node
                assert node in path_reward, "Error! node reward not recorded."
                r = path_reward[node] + self.V_hat[child]
                '''
                Iteratively update std, which is supposed to be faster.
                Population std.
                '''
                self.Q[node] += r
                old_ave_Q = self.ave_Q[node]
                self.ave_Q[node] = self.Q[node] / self.N[node]
                self.std[node] = self.sigma_0 if self.N[node] == 1 else sqrt(
                    ((self.N[node]-1)*self.std[node]**2 + (r - old_ave_Q) * (r - self.ave_Q[node]))/self.N[node])
                self.exploration_weight = max(
                    self.exploration_weight, abs(r))

    def _uct_select(self, node):
        "Select a child of node, balancing exploration & exploitation"

        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        all_actions = self.children[node]

        log_N_vertex = log(sum([self.N[c] for c in all_actions]))

        def uct(n):
            "Upper confidence bound for trees"
            return self.ave_Q[n] + self.exploration_weight * sqrt(
                2 * log_N_vertex / self.N[n]
            )

        return max(all_actions, key=uct)

    def _ocba_select(self, node):
        # All children of node should already be expanded:
        assert all(n in self.children for n in self.children[node])
        assert len(self.children[node]
                   ) > 0, "Error! Empty children action set!"

        if len(self.children[node]) == 1:
            return list(self.children[node])[0]

        all_actions = self.children[node]
        b = max(all_actions, key=lambda n: self.ave_Q[n])
        best_Q = self.ave_Q[b]
        suboptimals_set, best_actions_set = set(), set()
        for k in all_actions:
            if self.ave_Q[k] == best_Q:
                best_actions_set.add(k)
            else:
                suboptimals_set.add(k)
        delta = defaultdict(int)
        for k in all_actions:
            delta[k] = abs(self.ave_Q[k] - best_Q)

        if len(suboptimals_set) == 0:
            return min(all_actions, key=lambda n: self.N[n])
        # Choose a random one as reference
        ref = next(iter(suboptimals_set))

        para = defaultdict(int)
        ref_std_delta = self.std[ref]/delta[ref]
        para_sum = 0
        for k in suboptimals_set:
            para[k] = ((self.std[k]/delta[k])/(ref_std_delta))**2

        for k in best_actions_set:
            para[k] = sqrt(
                sum(
                    (self.std[k]*para[c]/self.std[c])**2 for c in suboptimals_set
                )
            )

        para_sum = sum(para.values())
        para[ref] = 1
        totalBudget = sum([self.N[c] for c in all_actions])+1

        ref_sol = (totalBudget)/para_sum

        return max(all_actions, key=lambda n: para[n]*ref_sol - self.N[n])


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
        return 1

    @abstractmethod
    def __eq__(node1, node2):
        "Nodes must be comparable"
        return True
