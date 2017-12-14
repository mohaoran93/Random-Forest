from Source.Data import Data
from Source.C45Tree import C45Tree
import pandas as pd
import numpy as np

# Split data set
df = pd.read_csv('./Data/car.csv')
# df = pd.read_csv('./Data/connect-4.csv')
# df = pd.read_csv('./Data/tic-tac-toe.csv')
# df = pd.read_csv('./Data/kr-vs-kp.csv')
msk = np.random.rand(len(df)) < 0.8
train_data = df[msk]
test_data = df[~msk]

train_data = Data(name=train_data)
test_data = Data(name=test_data)
# train

train_data.addSampleFromDf()
test_data.addSampleFromDf()

Tree = C45Tree(train_data)
print("===Train===")
Tree.train()
print("===Done===")

print("\n,Test")

T = 0
F = 0
for sample in test_data.getSamples():
    # if sample.getLabel() != 'unacc':
    classified = Tree.classify(sample)
    print("Compare: ",sample.getLabel(), classified)
    if sample.getLabel() == classified:
        T = T+1
    else:
        F = F+1
print(T/(T+F))

# experiments part




#attr = ['a1','a2']
#print(train[attr])
