
import pandas as pd
import numpy as np

class sample(object):
    """
    sample is an instance of data set
    """
    def __init__(self,feature,lable):
        self.features = feature
        self.lable = lable

    def setFeature(self,dic):
        self.features = dic
    def setLable(self,lable):
        self.lable = lable

    # def splitLeft(self, attributeNumber, threshold):
    #     if self.features[attributeNumber] <= threshold:
    #         return True
    #     return False
    def getValueAtIndex(self,index):
        return self.features[index]
    def getLabel(self):
        return self.lable
    def getFeature(self):
        return self.features
class Data(object):

    def __init__(self, name=None,data=None,index_value_tuple=None):
        """
        :param name: name can be DataFrame, None, name of a cvs file
                data: a list of samples
        :return:
        """
        self.index_value_tuple = index_value_tuple
        self.df = pd.DataFrame
        self.statistics = {}
        self.entropy = None
        if data == None:
            self.Data = [] # data is a list of samples
        else:
            self.Data = data
        # TODO now, I feel it is chaos
        if type(name)==pd.DataFrame:
            self.df = name
            self.DataName = 'temp_name'
        else:
            if name == None:
                name = input("Type in the data set file name then continue, {bscale, car, kr-vs-kp}")
                try:
                    self.df = pd.read_csv('./Data/'+name+'.csv')#'../Data/'
                    print('OK','I got the file: '+name+".csv")
                except FileNotFoundError:
                    print("No this file: "+name)
            else:
                self.DataName = name
    def addSample(self,sample):
        self.Data.append(sample)

    def addSampleFromDf(self):
        atts = [att for att in list(self.df) if att not in ['class']]
        #print(self.df.loc[:,atts])

        for r in range(len(self.df)):
            features = list(self.df.iloc[r][atts])
            sample_ = sample(features,self.df.iloc[r]['class'])
            self.Data.append(sample_)

    def isPure(self):
        count = 0
        standard = self.Data[0].getLabel()
        for s in self.Data:
            if s.getLabel() == standard:
                count = count+1
        print(count,len(self.Data))
        return count == len(self.Data)

    def PurePercentage(self):
        labes = {}
        for sample in self.Data:
            if sample.getLabel() not in labes.keys():
                labes[sample.getLabel()] = 1
            else:
                labes[sample.getLabel()] +=1
        popular_Label = max(labes,key=labes.get)
        num = labes[popular_Label]
        return (popular_Label,num/len(self.Data))
    def get_index_value_tuple(self):
        return self.index_value_tuple
    def getFirstSample(self):
        return self.Data[0]
    def getNumOfInstanceForLabel(self):
        """

        :return return the number of samples for each label :
        """
        #TODO if self.statistics is not empty
        labelToNum ={}
        for instance in self.Data:
            label = instance.getLabel()
            if label in labelToNum.keys():
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
        standard = self.Data[0].getLabel()
        for s in self.Data:
            if s.getLabel() == standard:
                count = count+1
        return count
    def getEntropy(self):

        # if self.entropy is not None:
        #     return self.entropy

        dic = self.getNumOfInstanceForLabel()
        total = len(self.Data)
        entropy = 0.0
        for label in dic.keys():  # TODO dic.keys()
            pOfclass = float(dic[label]) / float(total)
            if pOfclass != 0:
                entropy += pOfclass * np.log2(pOfclass)
        self.entropy = -entropy
        return self.entropy

    def splitBy(self, featureIndex):
        '''
        :param featureIndex:
        :return: a list of Data, each element has the same value on this featureIndex
        '''
        children_value = []
        children = {} # children[children_value,[sample_list]]
        children_obj = [] # list of Data samples
        for sample in self.Data:
            value = sample.getValueAtIndex(featureIndex)
            #if value not in children_value:
            if value not in children.keys():
                children_value.append(value)
                children[value] = [sample]
            else:
                children[value].append(sample)
        children_size = len(children)
        for i in range(children_size):
            value = list(children.keys())[i]
            children_obj.append(Data('children' + str(i),children[value],index_value_tuple=(featureIndex,value)))
            # TODO need to stash value to particular children
        return children_obj
    def splitOn(self, attributeNumber, threshold):

        left = []
        right = []

        for elem in self.Data:
            if elem.splitLeft(attributeNumber, threshold):
                left.append(elem)
            else:
                right.append(elem)

        leftData = Data("left", left)
        rightData = Data("right", right)

        return (leftData, rightData)

    def betterThreshold(self, feature):
        #Calculate the average value, split on that.
        totalN = len(self.Data)
        runningTotal = 0.0

        for samp in self.Data:
            runningTotal = runningTotal + samp.getValueAtIndex(feature)
        return float(runningTotal) / totalN

    def getSamplesOfFeature(self,featureIndex): # This function is dated
        children_value = []
        children = {} # children[children_value,[sample_list]]
        for sample in self.Data:
            value = sample.getValueAtIndex(featureIndex)
            if value not in children_value:
                children_value.append(value)
                children[value] = [sample]
            else:
                children[value].append(sample)
        return children

    def getDataLength(self):
        return len(self.Data)
    def getFeatureLength(self):
        return len(self.Data[0].getFeature())
    def getSamples(self, index=1):
        return self.Data
    def getSampleByIndex(self,index=0):
        return self.Data[index]
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
