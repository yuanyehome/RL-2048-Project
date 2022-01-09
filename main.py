import numpy as np
from os.path import join
import random
import argparse

from Agent import Agent, load_model
from draw import draw_score, draw_value

EPOCH_SIZE = 50000
MAX_NUM = 14 
ACTION_NUM = 4 # up, down, left, right
NAME = '8_tuple'
squeeze = False
merge = False
if squeeze:
    NAME += '_squeeze'
if merge:
    NAME += '_merge'

'''PATTERNS =  [
    [0,1,2,3],
    [1,5,9,13],
    [0,1,4,5],
    [1,2,5,6],
    [5,6,9,10]
]'''

'''MERGE_PATTERNS = [
    [0,2],
    [2,3],
    [3,4],
    [1,2],
    [1,3]
]'''

'''PATTERNS =  [
    [0,1,2,3,4,5],
    [0,1,2,4,5,6],
    [1,2,5,6,9,10],
    [0,1,4,5,9,13],
    [1,2,5,6,9,13]
]'''

PATTERNS = [
    [0,1,2,3,4,5,6,7],
    [0,1,2,3,4,5,8,12],
    [0,1,2,4,5,6,8,9],
    [1,2,5,6,9,10,13,14]
]

MERGE_PATTERNS = None

def make_100_mean_score(scores):
    scores = np.array(scores)
    scores = [np.mean(scores[i:i+100]) for i in range(0, len(scores), 100)]
    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--models", type=str, nargs="+")
    args = parser.parse_args()

    if args.train:
        print("Start training.")
        random.seed(2021)
        agent = Agent(NAME, PATTERNS, MERGE_PATTERNS, MAX_NUM, merge, squeeze)
        agent.train(EPOCH_SIZE)

    if args.eval:
        print("Start eval. Models:\n\t%s" % str(args.models))
        agents = []
        names = args.models
        for name in names:
            agents.append(load_model(name))

        scores = [make_100_mean_score([metric[0] for metric in agent.metrics]) for agent in agents]
        winning_rates = [agent.static['winning_rate'] for agent in agents]
        draw_score(names, winning_rates, 1000, "results")
