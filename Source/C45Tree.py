import random
import numpy as np

class TreeNode:
    def __init__(self, dataSet, featureList,roughValue = 1,parent=None,F = None,value = None):
        self.featureNumber = None  #This is the trained index of the feature to split on
        self.dataSet = dataSet
        self.featureList = featureList

        self.value = value # value of Particular attribute
        self.AttrValue = [] # (featureNumber,value) categories are corresponding to threshold
        self.children = [] # the list of TreeNodes
        self.parent = parent

        self.F = F
        self.roughValue = roughValue

    def c45Train(self,index_value=None):
        PP = self.dataSet.PurePercentage()
        roughValue = self.roughValue
        roughValue = max([roughValue,0.75])
        if PP[1] >= roughValue:
            label = self.dataSet.getSamples()[0].getLabel()
            leaf = LeafNode(label,value=index_value[1])
            leaf = LeafNode(PP[0],value=index_value[1])
            return leaf

        currentLength = self.dataSet.getDataLength() # instance size

        if self.F == None:
            Fvalue = len(self.featureList)
        else:
            Fvalue = min([self.F,len(self.featureList)]) # To avoid the give F is too large
        featureSubset = random.sample(self.featureList, Fvalue)
        maxGain = -1 * float("inf")
        H = self.dataSet.getEntropy()
        Goodchildren_dataSet = []
        for featureIndex in featureSubset:
            newEntropy = 0
            splitInfo = 0
            childrenList_dataSet = self.dataSet.splitBy(featureIndex)
            for child in childrenList_dataSet:
               PvalueOfA = child.getDataLength()/currentLength
               newEntropy = newEntropy + (PvalueOfA)*child.getEntropy()
               if PvalueOfA != 0:
                   splitInfo = splitInfo + (PvalueOfA)*np.log2(PvalueOfA)
            Gain = H - newEntropy
            if splitInfo == 0:
                Gain = 0 # TODO what is wrong?
            else:
                Gain = Gain/(-splitInfo)
            # if splitInfo == 0:
            #     print('splitInfo is 0, and Gain is ',Gain,len(childrenList_dataSet))
            # TODO Gain/splitInfo
            if Gain > maxGain:
                maxGain = Gain
                bestfeatureIndex = featureIndex
                Goodchildren_dataSet = childrenList_dataSet
        if len(Goodchildren_dataSet) == 0: # TODO it was 0
            # print("Never")
            dic = self.dataSet.getNumOfInstanceForLabel() # TODO
            bestLabel = None
            mostTimes = 0

            for label_ in dic:
                # print("leaf, key:", label_)
                if dic[label_] > mostTimes:
                    bestLabel = label_
                    mostTimes = dic[label_]
            leaf = LeafNode(bestLabel)
            return leaf
        else:
            newFeatureList = list(self.featureList)
            newFeatureList.remove(bestfeatureIndex)
            self.featureNumber = bestfeatureIndex

        for child in Goodchildren_dataSet:
            toTrain  = TreeNode(dataSet=child,featureList=newFeatureList,parent=self,F=self.F,value=child.get_index_value_tuple()[1])
            self.AttrValue.append(child.get_index_value_tuple())
            self.children.append(toTrain.c45Train(index_value=child.get_index_value_tuple())) #dataSet, featureList,
        return self
        
    def __str__(self):
        return str(self.featureList)

    def __repr__(self):
        return self.__str__()
                
    def classify(self, sample):
        value = sample.getValueAtIndex(self.featureNumber)
        for child in self.children:
            if child.getValue() ==value:#in [value,'leaf']:
                return child.classify(sample)
        # print("wrong!",value,self.featureNumber)
    def getAttrValue(self):
        return self.AttrValue
    def getValue(self):
        return self.value
    def getfeatureNumber(self):
        return self.featureNumber
    def setCategory(self,category):
        self.AttrValue = category

    def getdataSet(self):
        return self.dataSet


class LeafNode:
    '''
    A leaf node is a node that just has a classification 
    and is used to cap off a tree.
    '''

    def __init__(self, classification,value=None): # even leaf node needs the value of particular attribute
        self.classification = classification
        self.value = value
    def getAttrValue(self):
        return 'leaf'
    def getValue(self):
        return self.value
        # return 'leaf'

    def classify(self, sample):
        return self.classification


class C45Tree(object):

    def __init__(self, data,F = None,roughValue = 1):
        self.roughValue = roughValue
        self.rootNode = None
        self.data = data
        self.F = F
        self.FL = {}

    def train(self):
        length = self.data.getFeatureLength()
        featureIndices = range(length)
        self.rootNode = TreeNode(self.data, featureIndices,F=self.F,value=None)
        self.rootNode.c45Train()

    def getfeatureNumber(self):
        print(self.rootNode.getfeatureNumber())
    def classify(self, sample):
        '''
        Classify a sample based off of this trained tree.
        '''

        return self.rootNode.classify(sample)


