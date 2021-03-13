

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import dill
import argparse


def lineplot_pred(x_data, x_label, y1_data, y2_data, y_label, title, opp_policy, filetitle):
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
    ax1.set_title(title)

    # Create the second plot object, telling matplotlib that the two
    # objects have the same axis
    ax2 = ax1
    ax2.plot(x_data, y2_data, color=y2_color,
             linestyle='solid', marker='o', label='OCBA')
    ax1.set_ylim([0.65, 1])
    # Display legend
    ax1.legend(loc='lower right')
    plt.savefig(filetitle, format='eps')


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
    par1.axis['right'] = par1.get_grid_helper().new_fixed_axis(
        loc='right', axes=par1, offset=(0, 0))
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
    p2, = par1.plot(actions, ave_std, label="std",
                    linestyle='dashed', marker='v')
    p3 = par2.bar(actions, ave_N, label="# visits", color='cyan')

    # Set legend location
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
        '--checkpoint', type=str, help='relative path to checkpoint', default='ckpt/tic_tac_toe_uct_opponent_setup1.pkl')
    args = parser.parse_args()
    ckpt = args.checkpoint

    dill.load_session(ckpt)

    uct_ave_Q = uct_ave_Q_list[-1]
    ocba_ave_Q = ocba_ave_Q_list[-1]
    uct_ave_std = uct_ave_std_list[-1]
    ocba_ave_std = uct_ave_std_list[-1]
    uct_visit_cnt = uct_visit_ave_cnt_list[-1]
    ocba_visit_cnt = ocba_visit_ave_cnt_list[-1]

    if opp_first_move == 0:
        setup = 1
        actions = [i for i in range(1, 9)]
    elif opp_first_move == 4:
        setup = 2
        actions = list(range(0, 4)) + list(range(5, 9))
    else:
        raise ValueError('opponent first move should be either 0 or 4')
    uct_ave_Q_to_list, uct_ave_std_to_list, uct_ave_N_to_list = [], [], []
    ocba_ave_Q_to_list, ocba_ave_std_to_list, ocba_ave_N_to_list = [], [], []

    def sort_key(n):
        return sum([i for i in range(9) if n.state[i] == 1])
    for c in sorted(uct_ave_Q.keys(), key=sort_key):
        uct_ave_Q_to_list.append(uct_ave_Q[c])
        uct_ave_std_to_list.append(uct_ave_std[c])
        uct_ave_N_to_list.append(uct_visit_cnt[c])

    for c in sorted(ocba_ave_Q.keys(), key=sort_key):
        ocba_ave_Q_to_list.append(ocba_ave_Q[c])
        ocba_ave_std_to_list.append(ocba_ave_std[c])
        ocba_ave_N_to_list.append(ocba_visit_cnt[c])

    allocation_dist_plot(
        actions=actions,
        ave_Q=uct_ave_Q_to_list,
        ave_std=uct_ave_std_to_list,
        ave_N=uct_ave_N_to_list,
        title='results/TTT_sample_distribution_{opp_policy}_opponent_uct_setup{setup}.eps'.format(
            opp_policy=opp_policy, setup=setup)
    )
    allocation_dist_plot(
        actions=actions,
        ave_Q=ocba_ave_Q_to_list,
        ave_std=ocba_ave_std_to_list,
        ave_N=ocba_ave_N_to_list,
        title='results/TTT_sample_distribution_{opp_policy}_opponent_ocba_setup{setup}.eps'.format(
            opp_policy=opp_policy, setup=setup)
    )

    lineplot_pred(budget_range, 'N', results_uct, results_ocba,
                  'PCS', '', opp_policy=opp_policy, filetitle='results/tic_tac_toe_{opp_policy}_opponent_setup{setup}.eps'.format(opp_policy=opp_policy, setup=setup))
