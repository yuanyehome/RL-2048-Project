import numpy as np
import pickle
import os
import time
import random
from os.path import join
from World import Board, STOP_REWARD
from tqdm import tqdm

MAX_VALUE = 1<<31

def save_model(model, filepath):
    with open(join(filepath, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    return join(filepath, 'model.pkl')
    
def load_model(model_path):
    path = join(model_path, 'model.pkl')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def find_isomorphic_pattern(pattern):
    sample = Board(list(range(16)))
    isomorphic_pattern = []
    for i in range(8):
        if (i >= 4):
            board = Board(sample.mirror().tile )
        else:
            board = Board(sample.tile )
        for _ in range(i%4):
            board = board.rotate()
        isomorphic_pattern.append(np.array(board.tile)[pattern])
    return isomorphic_pattern

class TuplesNet():
    def __init__(self, pattern, maxnum, squeeze = False):
        self.V = np.zeros(([maxnum]*len(pattern)))
        self.pattern = pattern
        self.isomorphic_pattern = find_isomorphic_pattern(self.pattern)
        self.use_squeeze = squeeze
    
    def squeeze(self, pattern):
        new_pattern = pattern[1:]-pattern[:-1]
        new_pattern[new_pattern > 0] *= 2
        new_pattern = np.abs(new_pattern)
        new_pattern[new_pattern > 8] = 8
        return new_pattern

    def getState(self, tile):
        if self.use_squeeze:
            return [tuple(self.squeeze(np.array(tile)[p])) for p in self.isomorphic_pattern]
        return [tuple(np.array(tile)[p]) for p in self.isomorphic_pattern]
    
    def getValue(self, tile):
        S = self.getState(tile)
        V = [self.V[s] for s in S]
        V = sum(V) # / len(V)
        return V
    
    def setValue(self, tile, v, reset=False):
        S = self.getState(tile)
        v /= len(self.isomorphic_pattern)
        V = 0.0
        for s in S:
            self.V[s] = (v + self.V[s]) if not reset else v
            V += self.V[s]
        return V

class Agent():
    def __init__(self, name, patterns, merge_patterns = None, maxnum = 14, merge=False, squeeze = False):
        self.patterns = patterns
        self.Tuples = []
        self.maxnum = maxnum
        self.squeeze = squeeze
        for p in patterns:
            self.Tuples.append(TuplesNet(p, maxnum, squeeze))
        self.merged = not merge
        self.merge_patterns = merge_patterns
        self.metrics = []
        self.static = {'score':[], 'winning_rate':[]}
        self.max_mean_score = 0
        self.name = name
        self.filepath = join('models', name)
        self.run_time = 0.0
        self.start_time = 0.0
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
    
    def mergeTuple(self, t1:TuplesNet, t2:TuplesNet):
        p1 = t1.pattern
        p2 = t2.pattern
        p = [i for i in range(16) if (i in p1) or (i in p2)]
        dim = len(p)
        Tuple = TuplesNet(p, self.maxnum, self.squeeze)
        V1 = t1.V
        for idx, pos in enumerate(p):
            if pos not in p1:
                V1 = np.expand_dims(V1,idx).repeat(self.maxnum, axis=idx)
        V2 = t2.V
        for idx, pos in enumerate(p):
            if pos not in p2:
                V2 = np.expand_dims(V2,idx).repeat(self.maxnum, axis=idx)
        Tuple.V = V1+V2
        return Tuple

    def mergeTuples(self):
        if self.merged or self.merge_patterns == None:
            return
        new_Tuples = []
        for pair in self.merge_patterns:
            new_Tuples.append(self.mergeTuple(self.Tuples[pair[0]], self.Tuples[pair[1]]))
        self.Tuples = new_Tuples
        return

    def getValue(self, tile):
        V = [t.getValue(tile) for t in self.Tuples]
        V = sum(V) #/ len(V)
        return V
    
    def setValue(self, tile, v, reset=False):
        v /= len(self.Tuples)
        V = 0.0
        for t in self.Tuples:
            V += t.setValue(tile, v, reset)
        return V
    
    # get all s'
    def evaulate(self, next_games):
        #  r + V(s')
        return [(ng[1] + self.getValue(ng[0].tile))-MAX_VALUE*int(ng[1]<0) for ng in next_games]
    
    def learn(self, records, lr):
        exact = 0.0
        # (s, a, r, s', s'')
        for s, a, r, s_, s__ in records: 
            # V(s') = V(s') + \alpha ( r_next + V(s'_next) - V(s') )
            error = exact - self.getValue(s_)
            exact = r + self.setValue(s_, lr*error)
            
    def Stattistic(self, epoch, unit, show=True):
        f = open(join(self.filepath, 'log.txt'),'a+')
        metrics = np.array(self.metrics[epoch-unit:epoch])
        score_mean = np.mean(metrics[:, 0])
        score_max = np.max(metrics[:, 0])
        
        if show:
            f.write('\nepoch: {:<8d} time: {:>8.0f} Seconds\nmean = {:<8.0f} max = {:<8.0f}\n\n'.format(epoch, time.time() - self.start_time, score_mean, score_max))
        if (metrics.shape[1] < 3 or (epoch/unit)%10 != 0):
            f.close()
            return score_mean, score_max
        # all end game board
        metrics = np.array(self.metrics[epoch-10*unit:epoch])
        score_mean = np.mean(metrics[:, 0])
        score_max = np.max(metrics[:, 0])
        end_games = metrics[:, 2]
        reach_nums = np.array([1<<max(end) & -2 for end in end_games])
                  
        score_stat = []
        self.static['winning_rate'].append(0.0)
        for num in np.sort(np.unique(reach_nums)):
            # count how many game over this num
            reachs = np.count_nonzero(reach_nums >= num)
            reachs = (reachs*100)/len(metrics)
            # count how many game end at this num
            ends = np.count_nonzero(reach_nums == num)
            ends = (ends*100)/len(metrics)
            if show:
                f.write('{:<5d}  {:3.1f} % ({:3.1f} %)\n'.format(num, reachs, ends) )
            if num == 2048:
                self.static['winning_rate'][-1]=reachs
            score_stat.append( (num, reachs, ends) )
        
        score_stat = np.array(score_stat)
        self.static['score'].append(score_mean)
        f.close()
        if score_mean > self.max_mean_score:
            self.max_mean_score = score_mean
            self.run_time = time.time() - self.start_time
            save_model(self, self.filepath)
        elif not self.merged:
            self.mergeTuples()
            self.merged = True
            self.merge_eopch = epoch
        return score_mean, score_max, score_stat
    
    def train(self, epoch_size, lr=0.1, showsize=100):
        f = open(join(self.filepath, 'log.txt'),'w')
        f.write('# {}\n'.format(self.name))
        f.close()
        start_epoch = len(self.metrics)
        self.start_time = time.time() - self.run_time
        for epoch in tqdm(range(start_epoch, epoch_size)):
            # init score and env (2048)
            score = 0.0
            game = Board().popup().popup()
            records = []
            while True:
                # choose action
                next_games = [game.up(), game.down(), game.left(), game.right()]
                action = np.argmax(self.evaulate(next_games))
                next_game, reward = next_games[action]
                if game.end():
                    break
                next_game_after = next_game.popup()
                score += reward
                records.insert(0, (game.tile, action, reward, next_game.tile, next_game_after.tile) )
                game = next_game_after
                
            self.learn(records, lr)
            self.metrics.append((score, len(records), game.tile))
            if (epoch+1) % showsize == 0:
                self.Stattistic(epoch+1, showsize)

    def play(self, game):
        next_games = [game.up(), game.down(), game.left(), game.right()]
        action = np.argmax(self.evaulate(next_games))
        next_game, reward = next_games[action]
        return next_game, reward, ['up', 'down', 'left', 'right'][action]