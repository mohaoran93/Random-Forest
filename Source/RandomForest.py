from Source.C45Tree import *


class RandomForest:
    def __init__(self,data,F,NT):
        self.Data = data
        self.F = F
        self.NT = NT
        self.Trees = []
    def train(self):
        for i in range(self.NT):
            tree = C45Tree(self.Data,F=self.F)
            tree.train()
            self.Trees.append(tree)
        return self
    def classify(self,testData):
        print("NT:",len(self.Trees))
        T = 0
        F = 0
        for sample in testData.getSamples():
            values = {}
            for tree in self.Trees:
                value = tree.classify(sample)
                if value not in values.keys():
                    values[value] = 1
                else:
                    values[value] += 1
            voted = max(values,key=values.get)
            #print("compare: ",sample.getLabel() == voted)
            if sample.getLabel() == voted:
                T += 1
            else:
                F += 1
        acc = T/(T+F)
        print("acc:",acc)
               # print(tree.classify(sample))
