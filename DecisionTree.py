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
        chosenAttribute = chooseAttribute(dataset, predictiveAttributes)# Escolher atributo com o método de ganho de informação
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
        if len(df[label_column].unique()) == 1: 
            return True
        else: return False


tr = DecisionTree()
model = tr.train()