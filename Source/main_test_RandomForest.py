from Source.Data import Data
from Source.C45Tree import C45Tree
from Source.RandomForest import *
import pandas as pd
import numpy as np

# Split data set
# df = pd.read_csv('../Data/car.csv')
df = pd.read_csv('../Data/connect-4.csv')
# df = pd.read_csv('../Data/tic-tac-toe.csv')
# df = pd.read_csv('../Data/kr-vs-kp.csv')
msk = np.random.rand(len(df)) < 0.8
train_data = df[msk]
test_data = df[~msk]
train_data = Data(train_data)
test_data = Data(test_data)
train_data.addSampleFromDf()
test_data.addSampleFromDf()


M = test_data.getFeatureLength()
print(M)
sqrtM = int(np.ceil(np.sqrt(M)))
log2M = int(np.log2(M)+1)
pair = [(50,1),(50,3),(50,log2M),(50,sqrtM),(100,1),(100,3),(100,log2M),(100,sqrtM)]
for item in pair:
    NT = item[0]
    F = item[1]
    print("===Train===",NT,F)
    RF = RandomForest(train_data,F=F,NT=NT)
    RF.train()
    print("===Done===")
    RF.classify(testData=test_data)
#
