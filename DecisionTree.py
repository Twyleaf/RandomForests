from collections import namedtuple  
import math
import random
import pandas as pd

Node = namedtuple('Node', ['label', 'children'])

class DecisionTree():

    def train(self, dataset, predictiveAttributes, targetLabel):
        if self.datasetHasOnlyOneClass(dataset, targetLabel):# Se o dataset tiver apenas uma classe, criar um nó folha que prediz essa classe
            node = self.newNode(dataset[targetLabel].iloc[0])
            return node
        if len(predictiveAttributes) == 0:# Se atributos estiver vazio, retornar nodo com classe mais frequente
            node = self.newNode(self.getMostFrequentClass(dataset, targetLabel))
            return node
        chosenAttribute = self.chooseAttribute(dataset, predictiveAttributes, targetLabel)# Escolher atributo com o método de ganho de informação
        node = self.newNode(chosenAttribute)# Criar um nodo com esse atributo
        predictiveAttributes.remove(chosenAttribute)# Remover atributo escolhido da lista
        subsets = self.groupDatasetByAttributeValues(dataset, chosenAttribute)# Divide o dataset de acordo com cada valor diferente do atributo
        for subset in subsets:
            distinctValue = subset[0] # O valor distinto usado na divisão do dataset
            newDataset = subset[1] # O dataset dividido
            if newDataset.shape == 0:# Se subconjunto estiver vazio
                node = self.newNode(self.getMostFrequentClass(dataset, targetLabel))# Retornar nodo com classe mais frequente para o dataset
                return node
            else:
                # Criar um novo nodo e fazer uma chamada recursiva com o dataset dividido
                self.addChildren(father = node,
                                child = self.train(newDataset, predictiveAttributes.copy(), targetLabel),
                                transition = distinctValue)
        return node


    def newNode(self, label):
        return Node(label, [])

    def addChildren(self, father=None, child=None, transition=None):
        father.children.append( [child, transition] )

    def datasetHasOnlyOneClass(self, dataset, targetLabel):
        if len(dataset[targetLabel].unique()) == 1: 
            return True
        else:
            return False


    def entropy(self, dataset, targetLabel):
        s = dataset[targetLabel].value_counts(normalize=True) # Calcula as probabilidades de acordo com os valores alvos
        p = s.tolist()

        N = len(p)
        sum = 0

        for i in range(N):
            sum = sum + (p[i] * math.log2(p[i]))

        return -sum

    def calculateAttributesEntropy(self, dataset, predictiveAttributes, targetLabel):
        D = dataset.shape[0] # Número de linhas ou instâncias

        infoA = [] # Inicializa lista de entropia para cada atributo: (nome do atributo, entropia)

        attributesValues = [] # Lista de tuplas do tipo: (nome do atributo, lista de valores possíveis para o atributo)
        dict = {} # Dicionário que irá guardar os subconjuntos de cada valor possível de cada atributo

        for attribute in predictiveAttributes:
            s = dataset[attribute].unique() # Pega os valores únicos de cada coluna
            l = s.tolist()

            attributesValues.append((attribute, l)) # Salva uma tupla com nome do atributo e uma lista dos valores possíveis
            dict[attribute] = [] # Inicializa para fazer os cálculos de acordo com cada valor de cada atributo

        for tuple in attributesValues:
            attributeName = tuple[0]

            for value in tuple[1]: # Para cada valor na lista da tupla
                d = dataset[dataset[attributeName] == value] # Subconjunto do atributo com todas as instâncias com tal valor 
                dict[attributeName].append(d) # Salva o subconjunto

        for attribute in predictiveAttributes:
            sum = 0
            subsets = dict[attribute]

            for subset in subsets:
                proportion = subset.shape[0] / D
                sum = sum + (proportion * self.entropy(subset, targetLabel)) # Faz o cálculo da entropia restrito aos subconjuntos

            infoA.append((attribute, sum)) # Salva o valor do cálculo com nome do atributo

        return infoA

    def informationGain(self, dataset, infoA, targetLabel):
        infoD = self.entropy(dataset, targetLabel) # Entropia do dataset original
        infoGain = [] # Inicializa lista para calcular a diferença: (nome do atributo, diferença de entropias)

        for tuple in infoA:
            attributeName = tuple[0]
            infoAj = tuple[1]
            infoGain.append((attributeName, infoD - infoAj)) # Calcula o ganho estimado para cada atributo e salva

        return infoGain

    def chooseAttribute(self, dataset, predictiveAttributes, targetLabel):
        infoA = self.calculateAttributesEntropy(dataset, predictiveAttributes, targetLabel)
        infoGain = self.informationGain(dataset, infoA, targetLabel)
        chosenTuple = max(infoGain, key = lambda t: t[1])
        chosenAttribute = chosenTuple[0]

        print("Information gain: " + str(chosenTuple[1]) + ", " + chosenTuple[0])

        return chosenAttribute

    def groupDatasetByAttributeValues(self, dataset, chosenAttribute):
        s = dataset[chosenAttribute].unique() # Pega os valores únicos de cada coluna
        attributeValues = s.tolist()
        subsets = [] # Salva lista de tuplas: (valor, dataset)

        for value in attributeValues: # Cria um subconjunto para cada valor
            subset = dataset[dataset[chosenAttribute] == value]
            subsets.append((value, subset))

        return subsets

    def getMostFrequentClass(self, dataset, targetLabel):
        mostFrequent = dataset[targetLabel].value_counts().idxmax()

        return mostFrequent

    def printTree(self, node, index):
        
        if(index == 1):
            print('\n\tTree:')

        for i in range(0, index):
            print('\t', end='')
        for i in range(0, index):
            print('-', end='')

        print(node[0])

        if(len(node[1]) > 0):
            self.printTree(node[1][0][0], index + 1)
            self.printTree(node[1][1][0], index + 1)



# DT0 = DecisionTree()
# DT0Dataset = pd.read_csv("data/dadosBenchmark_validacaoAlgoritmoAD.csv", sep=';')
# DT0Predictive = list(DT0Dataset.columns)
# DT0Predictive.remove("Joga")
# DT0Target = "Joga"
# root = DT0.train(DT0Dataset, DT0Predictive, DT0Target)
# DTO.printTree(root, 1)

# DT1 = DecisionTree()
# DT1Dataset = pd.read_csv("data/house-votes-84.tsv", sep='\t')
# DT1Predictive = list(DT1Dataset.columns)
# DT1Predictive.remove("target")
# DT1Target = "target"
# root = DT1.train(DT1Dataset, DT1Predictive, DT1Target)
# DT1.printTree(root, 1)

# DT2 = DecisionTree()
# DT2Dataset = pd.read_csv("data/wine-recognition.tsv", sep='\t')
# DT2Predictive = list(DT2Dataset.columns)
# DT2Predictive.remove("target")
# DT2Target = "target"
# root = DT2.train(DT2Dataset, DT2Predictive, DT2Target)
# DT2.printTree(root, 1)

