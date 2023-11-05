import numpy as np
from IPython import display
import matplotlib.pyplot as plt

plt.ion()

def plot_progress(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()

    x = np.arange(1, len(scores) + 1)
    plt.plot(x, scores, label='Scores', marker='o', linestyle='-')
    plt.plot(x, mean_scores, label='Mean Scores', marker='x', linestyle='-')

    plt.plot(len(scores), scores[-1], 'ro')
    plt.plot(len(mean_scores), mean_scores[-1], 'rx')

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    plt.legend()
    plt.grid(True)

    plt.show()
    plt.pause(0.5)