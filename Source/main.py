from Source.Data import Data
from Source.C45Tree import C45Tree
from Source.RandomForest import *
import pandas as pd
import numpy as np
from Source.read import GetData

"""
dataset: you can select dataset from {car, tic-tac-toe, kr-vs-kp}

"""
train_data,test_data = GetData(name='kr-vs-kp').read()
M = test_data.getFeatureLength()
sqrtM = int(np.ceil(np.sqrt(M)))
log2M = int(np.log2(M)+1)
pair = [(50,1,1),(50,3,1),(50,log2M,1),(50,sqrtM,1),(100,1,1),(100,3,1),(100,log2M,1),(100,sqrtM,1),
        (50,1,0.9),(50,3,0.9),(50,log2M,0.9),(50,sqrtM,0.9),(100,1,0.9),(100,3,0.9),(100,log2M,0.9),(100,sqrtM,0.9)]
for item in pair:
    NT = item[0]
    F = item[1]
    R = item[2]
    # print("===Train=== NT = {0}, F = {1},roughValue = {2}".format(NT,F,R))
    RF = RandomForest(train_data,F=F,NT=NT,roughValue=1)
    RF.train()
    acc = RF.classify(testData=test_data)
    print("{0} & {1} & {2}& {3} & \\\\".format(NT,F,R,acc))
    # print("===Done===")
    dicF_V = RF.get_FeaturesList()
    print(dicF_V.keys())
#
