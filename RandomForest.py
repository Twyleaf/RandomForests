import pandas as pd
import random
from DecisionTree import DecisionTree

class RandomForest():

    def train(self, dataset, predictiveAttributes, targetLabel, numberOfTrees, bootstrapSize):
        listOfTrees = [] #inicializa lista de árvores
        for treeIndex in range(numberOfTrees):
            bootstrap = self.getBootstrap(dataset, bootstrapSize)
            decisionTree = DecisionTree()
            decisionTree.train(dataset, predictiveAttributes.copy(), targetLabel, True)
            ##decisionTree.printTree()
            listOfTrees.append(decisionTree)

        return listOfTrees

    def getBootstrap(self, dataset, bootstrapSize, replacement=True):
        return dataset.sample(bootstrapSize, replace=replacement)#retorna um bootstrap de tamanho bootstrapSize, com ou sem reposição


    def predict(self, listOfTrees, testDataset):
        listOfPredictions = []

        for tree in listOfTrees:
            prediction = tree.predictFromTrainingSet(testDataset)
            listOfPredictions.append(prediction)

        return listOfPredictions

    def voting(self, listOfPredictions, testDataset):
        h = 0

        for prediction in listOfPredictions: # Cria colunas no dataframe com as predições
            testDataset["h" + str(h)] = prediction
            h = h + 1

        predictionDataset = testDataset.loc[:, "h0":]
        votingResults = predictionDataset.mode(axis=1)[0]
        votingResultsList = votingResults.values.tolist()

        predictionDataset['voting'] = votingResultsList
        testDataset['voting'] = votingResultsList

        print(predictionDataset)
        print()
        print("Final prediction is", votingResultsList)

        return testDataset

"""
RF = RandomForest()
RFDataset = pd.read_csv("data/wine-recognition.tsv", sep='\t')
RFPredictive = list(RFDataset.columns)
RFPredictive.remove("target")
RFTarget = "target"
RF.train(RFDataset,RFPredictive,RFTarget,5,len(RFDataset))
"""

random.seed(10)
RF = RandomForest()
RFDataset = pd.read_csv("data/dadosBenchmark_treino.csv", sep=';')
RFDatasetTest = pd.read_csv("data/dadosBenchmark_teste.csv", sep=';')
RFPredictive = list(RFDataset.columns)
RFPredictive.remove("Joga")
RFTarget = "Joga"
listOfTrees = RF.train(RFDataset,RFPredictive,RFTarget,10,len(RFDataset))
listOfPredictions = RF.predict(listOfTrees, RFDatasetTest)
RF.voting(listOfPredictions, RFDatasetTest)


