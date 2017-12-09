from Source.Data import Data
from Source.C45Tree import C45Tree
import pandas as pd
import numpy as np

# Split data set
df = pd.read_csv('./Data/car.csv')
msk = np.random.rand(len(df)) < 0.8
train_data = df[msk]
test_data = df[~msk]

train_data = Data(train_data)
test_data = Data(test_data)
#size,dic = sample.getAttrOf()

# train

train_data.addSampleFromDf()
Tree = C45Tree(train_data)
Tree.train()
# experiments part




#attr = ['a1','a2']
#print(train[attr])
