import pandas as pd
from DecisionTree import DecisionTree

class RandomForest():

    def train(self, dataset, predictiveAttributes, targetLabel, numberOfTrees, bootstrapSize):
        listOfTrees = [] #inicializa lista de árvores
        for treeIndex in range(numberOfTrees):
            bootstrap = self.getBootstrap(dataset, bootstrapSize)
            decisionTree = DecisionTree()
            newTree = decisionTree.train(dataset, predictiveAttributes.copy(), targetLabel, True)
            ##decisionTree.printNode(newTree)
            listOfTrees.append(newTree)

        return listOfTrees

    def getBootstrap(self, dataset, bootstrapSize, replacement=True):
        return dataset.sample(bootstrapSize, replace=replacement)#retorna um bootstrap de tamanho bootstrapSize, com ou sem reposição


RF = RandomForest()
RFDataset = pd.read_csv("data/wine-recognition.tsv", sep='\t')
RFPredictive = list(RFDataset.columns)
RFPredictive.remove("target")
RFTarget = "target"
RF.train(RFDataset,RFPredictive,RFTarget,5,len(RFDataset))