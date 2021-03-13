from collections import namedtuple  
import math
import random
import pandas as pd

Node = namedtuple('Node', ['label', 'children'])



class DecisionTree():

    def train(self, dataset, predictiveAttributes, targetLabel):
        if self.datasetHasOnlyOneClass(dataset):# Se o dataset tiver apenas uma classe, criar um nó folha que prediz essa classe
            node = self.newNode(dataframe[targetLabel].iloc[0]) 
            return node
        if len(predictiveAttributes) == 0:# Se atributos estiver vazio, retornar nodo com classe mais frequente
            node = self._new_node(dataframe[targetLabel].mode()[0])
            return node
        chosenAttribute = chooseAttribute(dataset, predictiveAttributes, targetLabel)# Escolher atributo com o método de ganho de informação
        node = self.newNode(chosenAttribute)# Criar um nodo com esse atributo
        predictiveAttributes.remove(chosenAttribute)# remover atributo escolhido da lista
        for distinctValue, newDataset in self.groupDatasetByAttribute(dataset, chosenAttribute):#Para cada valor distinto do atributo no dataset
            if len(newDataset.index) == 0:# Se classe estiver vazia para esse valor
                node = self.newNode(getMostFrequentClass(dataset))# Retornar nodo com classe mais frequente para o dataset
                return node
            else:
                #Criar um novo nodo e fazer uma chamada recursiva com o dataset dividido
                self.addChildren(father = node,
                                child = self.train(newDataset, predictiveAttributes.copy(), targetLabel),
                                transition = distinctValue)
        return node


    def newNode(self, label):
        return Node(label, [])

    def addChildren(self, father=None, child=None, transition=None):
        father.children.append( [child, transition] )

    def datasetHasOnlyOneClass(dataset, targetLabel):
        if len(dataset[targetLabel].unique()) == 1: 
            return True
        else:
            return False


    def entropy(dataset, targetLabel):
        s = dataset[targetLabel].value_counts(normalize=True) # Calcula as probabilidades de acordo com os valores alvos
        p = s.tolist()

        N = len(p)
        sum = 0

        for i in range(N):
            sum = sum + (p[i] * math.log2(p[i]))

        return -sum

    def calculateAttributesEntropy(dataset, predictiveAttributes, targetLabel):
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
                sum = sum + (proportion * entropy(subset, targetLabel)) # Faz o cálculo da entropia restrito aos subconjuntos

            infoA.append((attribute, sum)) # Salva o valor do cálculo com nome do atributo

        return infoA

    def informationGain(dataset, infoA):
        infoD = entropy(dataset) # Entropia do dataset original
        infoGain = [] # Inicializa lista para calcular a diferença: (nome do atributo, diferença de entropias)

        for tuple in infoA:
            attributeName = tuple[0]
            infoAj = tuple[1]
            infoGain.append((attributeName, infoD - infoAj)) # Calcula o ganho estimado para cada atributo e salva

        return infoGain

    def chooseAttribute(dataset, predictiveAttributes, targetLabel):
        infoA = calculateAttributesEntropy(dataset, predictiveAttributes, targetLabel)
        infoGain = informationGain(dataset, infoA)
        chosenTuple = max(infoGain, key = lambda t: t[1])
        chosenAttribute = chosenTuple[0]

        return chosenAttribute


tr = DecisionTree()
model = tr.train()