import pandas as pd
import random
from DecisionTree import DecisionTree

def combineDatasets(datasets):
    for i, slice in enumerate(datasets):
        if i > 0: # Combina todas as divisões salvando no primeiro dataset
            datasets[0] = pd.concat([datasets[0], datasets[i]], ignore_index=True, sort=True)

    return datasets[0]

def kFoldSplit(k, dataset, targetLabel):
    chunkSize = int(dataset.shape[0] / k) # Calcula tamanho de um fold

    s = dataset[targetLabel].unique() # Pega os valores únicos do atributo alvo
    categories = s.tolist()

    datasetByCategories = []

    originalSize = [] # Para salvar número de instâncias para cada categoria
    originalProportions = [] # Para salvar a proporção de cada categoria
    foldNumberOfInstances = [] # Para salvar número de instâncias no fold para cada categoria

    for category in categories: # Divide o dataset original de acordo com as categorias do atributo alvo
        subset = dataset[dataset[targetLabel] == category]
        
        originalSize.append(len(subset))
        proportion = len(subset) / len(dataset)
        originalProportions.append(proportion)
        foldNumberOfInstances.append(int(proportion * chunkSize))
        datasetByCategories.append(subset)

    folds = []

    for i in range(k): # Para cada fold necessário
        fold = []

        for idx, nr in enumerate(foldNumberOfInstances): # Para cada número de instâncias necessárias para cada categoria
            fold.append(datasetByCategories[idx].iloc[0:nr, :]) # Salva até o número necessário
            datasetByCategories[idx] = datasetByCategories[idx].iloc[nr:-1, :] # Divide a partir do número necessário

        folds.append(combineDatasets(fold)) # Salva o fold combinado na lista de folds

    #for fold in folds:
    #    print(fold)

    return folds

class RandomForest():

    def train(self, dataset, predictiveAttributes, targetLabel, numberOfTrees, bootstrapSize):
        listOfTrees = [] #inicializa lista de árvores
        for treeIndex in range(numberOfTrees):
            bootstrap = self.getBootstrap(dataset, bootstrapSize)
            decisionTree = DecisionTree()
            decisionTree.train(dataset, predictiveAttributes.copy(), targetLabel, True)
            #decisionTree.printTree()
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
            testDataset["h" + str(h)] = pd.Series(prediction)
            h = h + 1

        predictionDataset = testDataset.loc[:, "h0":]
        votingResults = predictionDataset.mode(axis=1)[0]
        votingResultsList = votingResults.values.tolist()

        #predictionDataset["voting"] = votingResultsList
        testDataset["voting"] = votingResultsList

        #print("Final prediction is", votingResultsList)

        return testDataset

    def calculateAccuracy(self, targetLabel, testDataset):
        errors = testDataset[testDataset[targetLabel] != testDataset["voting"]]
        err = len(errors) / len(testDataset)

        return 1 - err

    def crossValidation(self, k, folds, predictiveAttributes, targetLabel, numberOfTrees):
        for i in range(k): # Cada fold deverá ser de teste
            testingDataset = folds[i]
            trainingFolds = []
            
            for j in range(k): # Para cada fold que não for de teste
                if i != j:
                    trainingFolds.append(folds[j]) # Combinar em um dataset

            trainingDataset = combineDatasets(trainingFolds)
            
            listOfTrees = self.train(trainingDataset, predictiveAttributes, targetLabel, numberOfTrees, len(trainingDataset))
            listOfPredictions = self.predict(listOfTrees, testingDataset)
            self.voting(listOfPredictions, testingDataset)
            print("ACCURACY #" + str(i + 1) + " is " + str(self.calculateAccuracy(targetLabel, testingDataset)))


random.seed(20)

"""
RF = RandomForest()
RFDataset = pd.read_csv("data/dadosBenchmark_treino.csv", sep=';')
RFDatasetTest = pd.read_csv("data/dadosBenchmark_teste.csv", sep=';')
RFPredictive = list(RFDataset.columns)
RFPredictive.remove("Joga")
RFTarget = "Joga"

listOfTrees = RF.train(RFDataset,RFPredictive,RFTarget,10,len(RFDataset))
listOfPredictions = RF.predict(listOfTrees, RFDatasetTest)
RF.voting(listOfPredictions, RFDatasetTest)
"""

RF = RandomForest()
RFDataset = pd.read_csv("data/house-votes-84.tsv", sep='\t')
RFPredictive = list(RFDataset.columns)
RFPredictive.remove("target")
RFTarget = "target"

k = 10
nTree = 10
folds = kFoldSplit(k, RFDataset, RFTarget)
RF.crossValidation(k, folds, RFPredictive, RFTarget, nTree)

"""
RF = RandomForest()
RFDataset = pd.read_csv("data/wine-recognition.tsv", sep='\t')
RFPredictive = list(RFDataset.columns)
RFPredictive.remove("target")
RFTarget = "target"

k = 10
nTree = 2
folds = kFoldSplit(k, RFDataset, RFTarget)
RF.crossValidation(k, folds, RFPredictive, RFTarget, nTree)
"""

