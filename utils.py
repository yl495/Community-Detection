import numpy as np
from sklearn.utils import shuffle
import math
import random

def loadData(labelfile, graphfile):
  data = []
  with open(labelfile) as inputfile:
      for line in inputfile:
          data.append(line.split())

  community1 = max(data, key = len)
  data.remove(community1)
  community2 = max(data, key = len)
  community1 = set(community1)
  community2 = set(community2)
  total_nodes = set(community1)
  for i in community2:
      total_nodes.add(i)

  total_nodes = list(total_nodes)

  graph = {}
  with open(graphfile) as inputfile:
      for line in inputfile:
          node = line.split()[0]
          neigh = line.split()[1]
          if (node not in community2 and node not in community1):
              continue
          if (neigh not in community2 and neigh not in community1):
              continue
          if node in graph:
              graph[node].add(neigh)
          else:
              graph[node] = {neigh}
          if neigh in graph:
              graph[neigh].add(node)
          else:
              graph[neigh] = {node}
  return community1, community2, graph

def preprocessingData(graph, community1, community2):
  adj_matrix = np.zeros((len(graph), len(graph)))
  degree_matrix = np.zeros((len(graph), len(graph)))
  node_index = {}
  j = 0
  for i in graph:
    node_index[i] = j
    j += 1
  for i in node_index:
    for j in graph[i]:
      if(j in node_index):
        adj_matrix[node_index[i]][node_index[j]] = 1
    degree_matrix[node_index[i]][node_index[i]] = len(graph[i])
  # generate one hot label
  label = np.zeros((len(graph), 3))
  for nodeId in node_index:
    if(nodeId in community1 and nodeId in community2):
      label[node_index[nodeId]] = [0, 0, 1]
    elif(nodeId in community2 and nodeId not in community1):
      label[node_index[nodeId]] = [0, 1, 0]
    else:
      label[node_index[nodeId]] = [1, 0, 0]
  return adj_matrix, degree_matrix, label


def trainData(label):
  n = label.shape[0]
  testingNum = math.floor(n*0.5)
  testIndex = random.sample(range(0, n), testingNum)
  training_label = np.zeros((n, 3))
  for i in range(n):
    training_label[i] = label[i]
  training_label[testIndex] = [None]
  return training_label, testIndex
    

def shuffleData(graph):
  key = list(graph.keys())
  random.shuffle(key)
  shuffle_graph = {k: graph[k] for k in key}
  return shuffle_graph

