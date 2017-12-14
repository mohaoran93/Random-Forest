# read the csv file
import pandas as pd
import numpy as np
from Source.Data import Data

class GetData():
    def __init__(self,name = None):
        self.name = name

    def read(self):
        name = self.name
        if name == None:
            name = input("Type in the data set file name then continue, {bscale, car, kr-vs-kp}")
        else:
            pass
        try:
            df = pd.read_csv('../Data/'+name+'.csv')
            print('OK','I got the file: '+name+".csv")
            msk = np.random.rand(len(df)) < 0.8
            train_data = df[msk]
            test_data = df[~msk]
            train_data = Data(train_data)
            test_data = Data(test_data)
            train_data.addSampleFromDf()
            test_data.addSampleFromDf()
            return (train_data,test_data)
        except:
            print("No this file")
            return None



