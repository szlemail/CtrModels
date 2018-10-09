import random
import numpy as np
import math
from LearnModel import LearnModel
from XgbModel import XgbModel
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
random.seed(2018)

class GA():
    def __init__(self, popcount, X, y, big_is_better = True, model = 'lgb'):
        
        # 种群数量
        self.count = popcount
        
        #生存模型
        if model == 'lgb':
            self.model = LearnModel(X, y)
        else:
            self.model = XgbModel(X, y)
        
        # 染色体长度 
        self.length = self.model.param_bit_length
        
        # 随机生成初始种群
        self.population = self.gen_population(self.length, popcount)
        
        self.score = dict()


    def evolve(self, retain_rate=0.3, random_select_rate=0.2, mutation_rate=0.1):
        """
        进化
        对当前一代种群依次进行选择、交叉并生成新一代种群，然后对新一代种群进行变异
        """
        parents = self.selection(retain_rate, random_select_rate)
        self.crossover(parents)
        self.mutation(mutation_rate)
        self.clearScore()

    def gen_chromosome(self, length):
        """
        随机生成长度为length的染色体，每个基因的取值是0或1
        这里用一个bit表示一个基因
        """
        chromosome = 0
        assert(length<64)
        for i in range(length):
            chromosome |= (1 << i) * random.randint(0, 1)
        return chromosome

    def gen_population(self, length, count):
        """
        获取初始种群（一个含有count个长度为length的染色体的列表）
        """
        return [self.gen_chromosome(length) for i in range(count)]

    def fitness(self, chromosome):
        """
        计算适应度，模型评分
        因为是求最大值，所以数值越大，适应度越高
        这里可以优化，parents 的基因没有变时，可以不用重新计算适应度。
        """
        if chromosome in self.score:
            #print("exist score:{} chromosome:{}".format(self.score[chromosome],chromosome))
            return self.score[chromosome]
        else:
            self.decode(chromosome)
            score = self.model.evalModel()
            self.score[chromosome] = score
            #print("new score:{} chromosome:{}".format(score,chromosome))
            return score
    
    def selection(self, retain_rate, random_select_rate):
        """
        选择
        先对适应度从大到小排序，选出存活的染色体
        再进行随机选择，选出适应度虽然小，但是幸存下来的个体
        """
        # 对适应度从大到小进行排序
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        fitscore = [x[0] for x in sorted(graded, reverse=True)]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        
        # 选出适应性强的染色体
        retain_length = int(len(graded) * retain_rate)
        parents = graded[:retain_length]
        
        # 轮盘赌方式，选出幸存的染色体
        fitscore = np.array(fitscore[retain_length:])
        if fitscore.max() > fitscore.min():
            fitscore = (fitscore - fitscore.min())/(fitscore.max() - fitscore.min())
        elif fitscore.max() < 0:
            fitscore = fitscore * (-1)
        fitscore = fitscore.cumsum()/fitscore.sum()
        player_count = len(fitscore)
        for i in range(int(self.count*random_select_rate)):#graded[retain_length:]:
            shot = random.random()
            left = 0
            right =player_count-1 
            index = int(player_count/2)
            if shot<=fitscore[left]:
                parents.append(graded[retain_length+left])
            else:
                while left<right-1:
                    if shot <= fitscore[index]:
                        right = index
                        index = int((left+index)/2)
                    else:
                        left = index
                        index = int((index+right)/2)
                parents.append(graded[retain_length+right])
                
                
        return parents

    def crossover(self, parents):
        """
        染色体的交叉、繁殖，生成新一代的种群
        """
        # 新出生的孩子，最终会被加入存活下来的父母之中，形成新一代的种群。
        children = []
        # 需要繁殖的孩子的量
        target_count = len(self.population) - len(parents)
        # 开始根据需要的量进行繁殖
        while len(children) < target_count:
            male = random.randint(0, len(parents)-1)
            female = random.randint(0, len(parents)-1)
#             difference = [ i^male for i in parents]
#             female = np.argmax(difference)
            if male != female:
                # 随机选取交叉点
                cross_pos = random.randint(0, self.length)
                cross_pos = self.model.getCross(cross_pos)
                # 生成掩码，方便位操作
                mask = 0
                for i in range(cross_pos):
                    mask |= (1 << i)
                male = parents[male]
                female = parents[female]
                # 孩子将获得父亲在交叉点前的基因和母亲在交叉点后（包括交叉点）的基因
                child = ((male & mask) | (female & ~mask)) & ((1 << self.length) - 1)
                if child not in parents:
                    if child not in children:
                        children.append(child)
        # 经过繁殖后，孩子和父母的数量与原始种群数量相等，在这里可以更新种群。
        self.population = parents + children

    def mutation(self, rate):
        """
        变异
        对种群中的所有个体，随机改变某个个体中的某个基因
        """
        for i in range(len(self.population)):
            if random.random() < rate:
                j = random.randint(0, self.length-1)
                self.population[i] ^= 1 << j

    def clearScore(self):
        keys = [*self.score.keys()]
        for chromosome in keys:
            if chromosome not in self.population:
                self.score.pop(chromosome)
                
    def showMaxScore(self):
        s = [i for i in self.score.values()]
        return(np.max(s))
                
    def decode(self, chromosome):
        self.model.decodeParam(chromosome)



    def predict(self,X):
        X = np.array(X)
        graded = [(self.fitness(chromosome), chromosome) for chromosome in self.population]
        graded = [x[1] for x in sorted(graded, reverse=True)]
        self.decode(graded[0])
        self.model.trainModel()
        return self.model.predict(X)
        
    def printParam(self):
        score = self.showMaxScore()
        for item in self.score.items():
            if item[1] == score:
                self.decode(item[0])
                break;
        print("max score = {:.4f}".format(score))
        print(self.model.printParams(score))
        return score
    
