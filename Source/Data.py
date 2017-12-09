
import pandas as pd
import numpy as np

class sample(object):
    """
    sample is an instance of data set
    """
    def __init__(self,feature,lable):
        self.feature = feature
        self.lable = lable

    def setFeature(self,dic):
        self.feature = dic
    def setLable(self,lable):
        self.lable = lable

    def getLabel(self):
        return self.lable
    def getFeature(self):
        return self.feature
class Data(object):

    def __init__(self, name=None):
        """
        :param name: name can be DataFrame, None, name of a cvs file
        :return:
        """
        self.df = pd.DataFrame
        self.data = [] # data is a list of samples
        self.statistics = {}
        self.entropy = None
        if type(name)==pd.DataFrame:
            self.df = name
        else:
            if name == None:
                name = input("Type in the data set file name then continue, {bscale, car, kr-vs-kp}")
            try:
                self.df = pd.read_csv('./Data/'+name+'.csv')#'../Data/'
                print('OK','I got the file: '+name+".csv")
            except FileNotFoundError:
                print("No this file: "+name)

    def addSample(self,sample):
        self.data.append(sample)

    def addSampleFromDf(self):
        atts = [att for att in list(self.df) if att not in ['class']]
        #print(self.df.loc[:,atts])

        for r in range(len(self.df)):
            features = list(self.df.iloc[r][atts])
            sample_ = sample(features,self.df.iloc[r]['class'])
            self.data.append(sample_)

    def isPure(self):
        return self.count() <= 1
    def getNumOfInstanceForLabel(self):
        """

        :return return the number of samples for each label :
        """
        #TODO if self.statistics is not empty
        labelToNum ={}
        for instance in self.data:
            label = instance.getLabel
            if label in labelToNum:
                labelToNum[label] += 1
            else:
                labelToNum[label] = 1
        self.statistics = labelToNum
        return self.statistics
    def count(self):
        """
        :return count : count is the number of instances that have the same class
        """
        count = 0
        standard = self.data[0].getLabel()
        for s in self.data:
            if s.getLabel() == standard:
                count = count+1
        return count
    def getEntropy(self):
        '''
        Returns a numerical quantity of the entropy for this
        data set.
        '''

        if self.entropy is not None:
            return self.entropy

        dic = self.getNumOfInstanceForLabel()

        #assuming all the data is labeled
        total = len(self.data)
        entropy = 0.0

        for label in dic:
            #Calculate the probability of the key
            pOfclass = float(dic[label]) / float(total)
            #Calculate the entropy for this key and add it to the running sum
            if pOfclass != 0:
                entropy = entropy + -1 * pOfclass * np.log2(pOfclass)

        self.entropy = entropy
        return self.entropy

    def getDataLength(self):
        return len(self.data)
    def getFeatureLength(self):
        return len(self.data[0].getFeature())
    def getSamples(self, index=1):
        return self.data
    def getSampleByIndex(self,index=0):
        return self.data[index]
    def getAttrSize(self,df=None):
        """
        :param df: optional
        :return attr_size, dic:
        example:
        attr_size: 6, the number of attr is 6
        dic: 'a1': array(['vhigh', 'high', 'med', 'low']
        """
        df = self.df
        attr_size = len(df.columns) - 1
        rows_size = len(df) # didn't use for the time beings
        dic = {}
        for c in list(df):
            if c != 'class':
              values = df[c].unique()
              dic[c] = values
        for c in np.arange(1,attr_size):
            c = 'c'+ str(c)
        return attr_size,dic
