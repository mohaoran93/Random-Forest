import random
import numpy as np
import sys

class TreeNode:
    def __init__(self, dataSet, featureList, parent=None):
        self.featureNumber = None  #This is the trained index of the feature to split on
        self.dataSet = dataSet
        self.featureList = featureList

        self.threshold = None     #This is the trained threshold of the feature to split on
        self.leftChild = None
        self.rightChild = None
        self.children = []
        self.parent = parent

    def c45Train(self):
        print(self.dataSet.isPure())
        if(self.dataSet.isPure()):
            #gets the label of the first data instance and makes a leaf node
            #classifying it. 
            label = self.dataSet.getSamples()[0].getLabel()
            leaf = LeafNode(label)
            return leaf
        #If there are no more features in the feature list
        if len(self.featureList) == 0:  # TODO
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

        #Check all of the features for the split with the most 
        #information gain. Use that split.
        currentEntropy = self.dataSet.getEntropy()  # done
        currentLength = self.dataSet.getDataLength() # instance size
        infoGain = -1 * float("inf")
        bestFeature = 0
        bestLeft = None
        bestRight = None
        bestThreshold = 0

        #Feature Bagging, Random subspace
        #TODO F will be a parameter for randome forest
        Fvalue = int(np.ceil(np.sqrt(len(self.featureList))))
        # index of featureSubset
        featureSubset = random.sample(self.featureList, Fvalue)
        print('***************','\n',"featureSubset: ",featureSubset)
        for featureIndex in featureSubset:
            maxGain = 0
            Goodchildren = []
            childrenList = self.dataSet.splitBy(featureIndex)
            H = self.dataSet.getEntropy()
            newEntropy = 0
            for child in childrenList:
               entropy = child.getEntropy()
               newEntropy = newEntropy + (child.getDataLength()/currentLength)*child.getEntropy()
            Gain = H - newEntropy
            # TODO Gain/splitInfo
            print("Gain:",Gain)
            if Gain > maxGain:
                maxGain = Gain
                bestfeatureIndex = featureIndex
                Goodchildren = childrenList


        if len(Goodchildren) == 0: # TODO
            print("len(Goodchildren) == 0")
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
            newFeatureList.remove(bestfeatureIndex)  # TODO when it does not find bestfeatureIndex
            print('newFeatureList: ',newFeatureList)

        #self.featureNumber = bestFeature
        #
        # leftChild = TreeNode(bestLeft, newFeatureList, self)
        # rightChild = TreeNode(bestRight, newFeatureList, self)
        #
        # self.leftChild = leftChild.c45Train()
        # self.rightChild = rightChild.c45Train()
        for child in Goodchildren:
            self.children.append(TreeNode(child,newFeatureList,self))
        for child in self.children:
            child.c45Train()
        return self
        
    def __str__(self):
        return str(self.featureList)

    def __repr__(self):
        return self.__str__()
                
    def classify(self, sample):
        '''
        Recursivly traverse the tree to classify the sample that is passed in. 
        '''

        value = sample.getFeatures()[self.featureNumber]

        if(value < self.threshold):
            #Continue down the left child    
            return self.leftChild.classify(sample)

        else:
            #continue down the right child
            return self.rightChild.classify(sample)


class LeafNode:
    '''
    A leaf node is a node that just has a classification 
    and is used to cap off a tree.
    '''

    def __init__(self, classification):
        self.classification = classification

    def classify(self, sample):
        #A leaf node simply is a classification, return that
        #This is the base case of the classify recursive function for TreeNodes
        return self.classification


class C45Tree(object):

    def __init__(self, data):
        self.rootNode = None
        self.data = data

    def train(self):
        '''
        Trains a decision tree classifier on data set passed in. 
        The data set should contain a good mix of each class to be
        classified.
        '''
        length = self.data.getFeatureLength()  # done
        featureIndices = range(length)
        self.rootNode = TreeNode(self.data, featureIndices)
        self.rootNode.c45Train()

    def classify(self, sample):
        '''
        Classify a sample based off of this trained tree.
        '''

        return self.rootNode.classify(sample)


