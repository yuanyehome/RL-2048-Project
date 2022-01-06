import matplotlib.pyplot as plt
import numpy as np

def draw_score(names, scores, showsize):
    plt.xlabel('Training games (x {})'.format(showsize))
    #plt.ylabel('score')
    plt.ylabel('winning rate')
    line_styles = ['-', '--', '-.', ':']
    x = range(max([len(score) for score in scores]))
    for idx, score in enumerate(scores):
        plt.plot(x[:len(score)],score,linestyle=line_styles[idx],label=names[idx])
    plt.legend()
    plt.savefig("score.png")
    plt.cla()


def draw_value(value):
    plt.xlabel('log10(idx/100)')
    plt.ylabel('value')
    x = np.array(range(len(value)))+1
    x = np.log10(x)
    plt.plot(x, value, linewidth = 1)
    plt.savefig("value.png")
    plt.cla()
