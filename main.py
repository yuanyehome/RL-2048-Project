import time
import numpy as np
from os.path import join
import pickle
import random

from World import Board, showstatus
from Agent import Agent, load_model
from draw import draw_score, draw_value

EPOCH_SIZE = 50000
MAX_NUM = 14 
ACTION_NUM = 4 # up, down, left, right
NAME = 'tt_4_2_tuple'
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
    [0],
    [0, 1],
    [1, 2],
    [4, 5],
    [5, 6]
]

MERGE_PATTERNS = None

def make_100_mean_score(scores):
    scores = np.array(scores)
    scores = [np.mean(scores[i:i+100]) for i in range(0, len(scores), 100)]
    return scores


if __name__ == '__main__':
    random.seed(2021)
    path = join('models', NAME)
    agent = Agent(NAME, PATTERNS, MERGE_PATTERNS, MAX_NUM, merge, squeeze)
    agent.train(EPOCH_SIZE)
    agents = []
    names = ['tt_4_3_tuple']
    for name in names:
        path = join('models', name)
        agents.append(load_model(path))
    
    scores = [make_100_mean_score([metric[0] for metric in agent.metrics]) for agent in agents]
    winning_rates = [agent.static['winning_rate'] for agent in agents]
    draw_score(names, winning_rates, 1000)
    '''
    Tuples = agent.Tuples
    Tuples_value = [Tuple.V.ravel() for Tuple in Tuples]
    Tuples_abs_value_distribution = [-np.sort(-np.abs(value)) for value in Tuples_value]
    draw_value(Tuples_abs_value_distribution[0][::100])
    '''
    '''
    game = Board().popup().popup()
    print(game.__str__())
    while game.end() == False:
        next_game, reward, action = agent.play(game)
        next_game = next_game.popup()
        print('reward:', reward)
        print('action:', action)
        print(next_game.__str__())
        game = next_game
    '''
    