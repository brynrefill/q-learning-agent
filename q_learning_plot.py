import matplotlib.pyplot as plt
import numpy as np

N_EPISODES = 500 # if is_det else 1000

def plot_data(data1, data2):
    # DATA
    data1 = np.array(data1).astype(np.int64)
    data2 = np.array(data2).astype(np.int64)

    # (array of) episode numbers (x-axis)
    episodes = np.arange(0, N_EPISODES)

    # optimal value
    optimal_value = np.full(N_EPISODES, -13)

    # --- CREATE THE PLOT ---
    plt.figure(figsize=(7, 4)) # (10, 6)

    # plot the lines
    plt.plot(episodes, optimal_value, color='#0072B2', linewidth=1, linestyle='--', label='Optimal value')
    plt.plot(episodes, data1, color='#E69F00', linewidth=1, label='Tabular Q-learning')
    plt.plot(episodes, data2, color='#009E73', linewidth=1, label='Deep Q-Network')

    # style the lines
    plt.xlabel('Episode', fontsize=11)
    plt.ylabel('Cumulative reward', fontsize=11)
    plt.title('100-Episode Moving Average of Cumulative Rewards', fontsize=12, fontweight='bold')

    # calculate y-axis min and max
    combined = np.concatenate((data1, data2))
    y_min = np.min(combined)
    y_max = 100

    # set the limits of the x-axis and y-axis on the plot
    plt.xlim(0, N_EPISODES)
    plt.ylim(y_min, y_max)

    # set x-axis and y-axis values
    plt.xticks(range(0, N_EPISODES + 1, 50)) # if is_det else 100
    
    # generate 6 evenly spaced tick marks
    # and make y-axis values neat
    yticks = np.linspace(y_min - 100, y_max, 6)
    yticks = sorted(list(np.round(yticks / 100) * 100) + [-13])
    # yticks = [t for t in yticks if t != 100] # if not is_det
    plt.yticks(yticks)
    plt.tick_params(axis='both', labelsize=10)

    # draw the figure once so tick labels exist and style the optimal value label
    plt.draw()
    for label in plt.gca().get_yticklabels():
        if label.get_text() == '−13':
            label.set_fontsize(8)
            label.set_fontweight('bold')
            label.set_color('#0072B2')

    # --- GRID ---
    # turn the grid on and control the transparency
    plt.grid(True, alpha=0.9)

    # add a legend to the plot and customize its position and font size
    plt.legend(loc='lower right', fontsize=10)

    # add background color
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')

    # automatically adjust the spacing, so everything fits nicely without overlapping
    plt.tight_layout()

    # save the plot to a file
    # control the resolution and trim extra whitespace around the figure
    plt.savefig('q_learning_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'q_learning_plot.png'")

    # plt.show()
