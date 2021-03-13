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
from monte_carlo_tree_search import MCTS, Node
from numpy import sqrt, savez
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import dill
import argparse


def lineplot_pred(x_data, x_label, y1_data, y2_data, y_label, title):
    # Each variable will actually have its own plot object but they
    # will be displayed in just one plot
    # Create the first plot object and draw the line
    y1_color = '#539caf'
    y2_color = 'red'
    _, ax1 = plt.subplots()
    ax1.plot(x_data, y1_data, color=y1_color,
             linestyle='dashed', marker='v', label="UCT")
    # Label axes
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same axis
    ax2 = ax1
    ax2.plot(x_data, y2_data, color=y2_color,
             linestyle='solid', marker='o', label='OCBA')
    ax1.set_ylim([0.6, 1])
    # Display legend
    ax1.legend(loc='lower right')
    plt.savefig(title+'.eps', format='eps')


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
    plt.subplots_adjust(right=0.75)

    par1 = host.twinx()
    par2 = host.twinx()

    offset = 50
    new_fixed_axis = par2.get_grid_helper().new_fixed_axis
    par1.axis['right'] = par1.get_grid_helper().new_fixed_axis(
        loc='right', axes=par1, offset=(0, 0))
    par2.axis["right"] = new_fixed_axis(loc="right",
                                        axes=par2,
                                        offset=(offset, 0))

    par2.axis["right"].toggle(all=True)

    host.set_xlim(actions[0]-0.5, actions[-1]+0.5)
    host.set_ylim(1.1*min(ave_Q), 0.7*max(ave_Q))
    par1.set_ylim(0, 1.1*max(ave_std))
    par2.set_ylim(0, 1.1*max(ave_N))

    host.set_xlabel("Actions")
    host.set_ylabel("Estimated value function Q")
    par1.set_ylabel("Estimated standard deviation")
    par2.set_ylabel("Average number of visits")

    p1, = host.plot(actions, ave_Q, label="Q", linestyle='solid', marker='o')
    p2, = par1.plot(actions, ave_std, label="std",
                    linestyle='dashed', marker='v')
    p3 = par2.bar(actions, ave_N, label="# visits", color='cyan')

    # Adjust lengend location here
    host.legend(loc=(0.7, 0.5))

    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.patches[0].get_facecolor())

    fig = plt.gcf()
    plt.draw()
    plt.show()
    fig.savefig(title, format='eps', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str, help='relative path to checkpoint', default='ckpt/Inventory_K0_p10_budget_1000.pkl')
    args = parser.parse_args()
    ckpt = args.checkpoint

    dill.load_session(ckpt)
    uct_ave_Q_to_list, uct_ave_std_to_list, uct_ave_N_to_list = [], [], []
    ocba_ave_Q_to_list, ocba_ave_std_to_list, ocba_ave_N_to_list = [], [], []
    actions = []
    for c in sorted(uct_mcts.children[uct_root_node], key=lambda n: n.action):
        uct_ave_Q_to_list.append(uct_ave_Q[c])
        uct_ave_std_to_list.append(uct_ave_std[c])
        uct_ave_N_to_list.append(uct_visit_cnt[c])

    for c in sorted(uct_mcts.children[uct_root_node], key=lambda n: n.action):
        ocba_ave_Q_to_list.append(ocba_ave_Q[c])
        ocba_ave_std_to_list.append(ocba_ave_std[c])
        ocba_ave_N_to_list.append(ocba_visit_cnt[c])
        actions.append(c.action)

    allocation_dist_plot(
        actions=actions,
        ave_Q=uct_ave_Q_to_list,
        ave_std=uct_ave_std_to_list,
        ave_N=uct_ave_N_to_list,
        title='results/Inventory_sample_distribution_uct_K{}_p{}.eps'.format(
            K, p)
    )
    allocation_dist_plot(
        actions=actions,
        ave_Q=ocba_ave_Q_to_list,
        ave_std=ocba_ave_std_to_list,
        ave_N=ocba_ave_N_to_list,
        title='results/Inventory_sample_distribution_ocba_K{}_p{}.eps'.format(
            K, p)
    )

    lineplot_pred(budget_range, 'N', results_uct, results_ocba,
                  'PCS', 'results/Inventory_K{}_p{}'.format(K, p))
