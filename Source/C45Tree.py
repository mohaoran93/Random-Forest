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

            for key in labels:
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

        for featureIndex in featureSubset:
            #Calculate the threshold to use for that feature
            threshold = self.dataSet.betterThreshold(featureIndex)  # TODO

            (leftSet, rightSet) = self.dataSet.splitOn(featureIndex, threshold)

            leftEntropy = leftSet.getEntropy()
            rightEntropy = rightSet.getEntropy()
            #Weighted entropy for this split
            newEntropy = (leftSet.getLength() / currentLength) * leftEntropy + (rightSet.getLength() / currentLength) * rightEntropy
            #Calculate the gain for this test
            newIG = currentEntropy - newEntropy

            if(newIG > infoGain):
                #Update the best stuff
                infoGain = newIG
                bestLeft = leftSet
                bestRight = rightSet
                bestFeature = featureIndex
                bestThreshold = threshold

        newFeatureList = list(self.featureList)
        newFeatureList.remove(bestFeature)

        #Another base case, if there are no good features to split on
        if bestLeft.getLength() == 0 or bestRight.getLength() == 0:
            labels = self.dataSet.getLabelStatistics()
            bestLabel = None
            mostTimes = 0

            for key in labels:
                if labels[key] > mostTimes:
                    bestLabel = key
                    mostTimes = labels[key]
            #Make the leaf node with the best label
            leaf = LeafNode(bestLabel)
            return leaf

        self.threshold = bestThreshold
        self.featureNumber = bestFeature

        leftChild = TreeNode(bestLeft, newFeatureList, self)
        rightChild = TreeNode(bestRight, newFeatureList, self)

        self.leftChild = leftChild.c45Train()
        self.rightChild = rightChild.c45Train()

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


