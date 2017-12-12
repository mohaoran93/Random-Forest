import random
import numpy as np
import sys

class TreeNode:
    def __init__(self, dataSet, featureList, parent=None):
        self.featureNumber = None  #This is the trained index of the feature to split on
        self.dataSet = dataSet
        self.featureList = featureList

        self.threshold = None     #This is the trained threshold of the feature to split on
        self.leftChild = None  # left and right child is used for continuous attributes
        self.rightChild = None

        self.AttrValue = [] # (featureNumber,value) categories are corresponding to threshold
        self.children = [] # the list of TreeNodes
        self.parent = parent

    def c45Train(self):
        if(self.dataSet.isPure()):
            #gets the label of the first data instance and makes a leaf node
            #classifying it. 
            label = self.dataSet.getSamples()[0].getLabel()
            leaf = LeafNode(label)
            print("this node is Pure: ",label)
            return leaf
        #If there are no more features in the feature list
        if len(self.featureList) == 0:  # TODO It Seems never get there
            labels = self.dataSet.getNumOfInstanceForLabel()
            bestLabel = None
            counts = 0
            print('there are no more features in the feature list')
            for key in labels: # TODO labes are dictionaries, there may be sth incorrect
                print(key)
                if labels[key] > counts:
                    bestLabel = key
                    counts = labels[key]
            #Make the leaf node with the label with largest size
            leaf = LeafNode(bestLabel)
            return leaf

        currentEntropy = self.dataSet.getEntropy()  # done
        currentLength = self.dataSet.getDataLength() # instance size

        #TODO F will be a parameter for randome forest
        Fvalue = int(np.ceil(np.sqrt(len(self.featureList))))
        Fvalue = len(self.featureList)

        featureSubset = random.sample(self.featureList, Fvalue)
        print('***************','\n',"featureSubset: ",featureSubset)
        maxGain = -1 * float("inf")
        for featureIndex in featureSubset:
            Goodchildren_dataSet = []
            childrenList_dataSet = self.dataSet.splitBy(featureIndex)

            H = self.dataSet.getEntropy()
            newEntropy = 0

            for child in childrenList_dataSet:
               newEntropy = newEntropy + (child.getDataLength()/currentLength)*child.getEntropy()
            Gain = H - newEntropy
            # TODO Gain/splitInfo
            print("Gain:",Gain)
            if Gain > maxGain:
                maxGain = Gain
                bestfeatureIndex = featureIndex
                Goodchildren_dataSet = childrenList_dataSet


        if len(Goodchildren_dataSet) == 0: # TODO
            print("len(Goodchildren_dataSet) == 0")
            dic = self.dataSet.getNumOfInstanceForLabel() # TODO
            bestLabel = None
            mostTimes = 0

            for label_ in dic:
                print("leaf, key:", label_)
                if dic[label_] > mostTimes:
                    bestLabel = label_
                    mostTimes = dic[label_]
            #Make the leaf node with the best label
            leaf = LeafNode(bestLabel)
            return leaf
        else:
            newFeatureList = list(self.featureList)
            newFeatureList.remove(bestfeatureIndex)
            self.featureNumber = bestfeatureIndex
            #print('newFeatureList: ',newFeatureList)

        for child in Goodchildren_dataSet:
            toTrain  = TreeNode(dataSet=child,featureList=newFeatureList,parent=self)
            self.AttrValue.append(child.get_index_value_tuple())
            self.children.append(toTrain.c45Train()) #dataSet, featureList,
        # for child in self.children:
        #     child.c45Train()
        return self
        
    def __str__(self):
        return str(self.featureList)

    def __repr__(self):
        return self.__str__()
                
    def classify(self, sample):
        '''
        Recursivly traverse the tree to classify the sample that is passed in. 
        '''
        print("children size: ", len(self.children))
        # for child in self.children:
        #     if child.getCategory() == 'leaf':
        #         print("Got leaf")
        #         return child.classify(sample)
        #     else:
        #         tuple_ = child.getCategory()  # get_index_value_tuple
        #         index = self.featureNumber
        #         value =
        #         if sample.getValueAtIndex(index=index) == value:
        #             #print("match index value pair")
        #             return child.classify(sample)
        for i in range(len(self.children)):
            if self.children[i].getCategory() == 'leaf':
                print("Got leaf")
                return self.children[i].classify(sample)
            else:
                index = self.featureNumber
                value = self.AttrValue[i]
                if sample.getValueAtIndex(index=index) == value:
                    #print("match index value pair")
                    return self.children[i].classify(sample)

    def getCategory(self):
        return self.AttrValue

    def setCategory(self,category):
        self.AttrValue = category

    def getdataSet(self):
        return self.dataSet


class LeafNode:
    '''
    A leaf node is a node that just has a classification 
    and is used to cap off a tree.
    '''

    def __init__(self, classification):
        self.classification = classification

    def getCategory(self):
        return 'leaf'

    def classify(self, sample):
        #A leaf node simply is a classification, return that
        #This is the base case of the classify recursive function for TreeNodes
        print("Got leaf: ",self.classification)
        return self.classification


class C45Tree(object):

    def __init__(self, data):
        self.rootNode = None
        self.data = data

    def train(self):

        length = self.data.getFeatureLength()
        featureIndices = range(length)
        self.rootNode = TreeNode(self.data, featureIndices)
        self.rootNode.c45Train()

    def classify(self, sample):
        '''
        Classify a sample based off of this trained tree.
        '''

        return self.rootNode.classify(sample)


