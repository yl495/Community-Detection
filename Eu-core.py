import tensorflow as tf
import numpy as np
communities = {}
with open('email-Eu-core-department-labels.txt') as inputfile:
  for line in inputfile:
    data = (line.split())
    if(data[1] not in communities):
      communities[data[1]] = [data[0]]
    else:
      communities[data[1]].append(data[0])
community1 = communities.pop(max(communities, key=lambda x:len(communities[x])))
community2 = communities.pop(max(communities, key=lambda x:len(communities[x])))
community3 = communities.pop(max(communities, key=lambda x:len(communities[x])))
community1 = set(community1)
community2 = set(community2)
community3 = set(community3)
with open('email-Eu-core.txt') as inputfile:
    graph = {}
    for line in inputfile:
        node = line.split()[0]
        neigh = line.split()[1]
        if(node in community1 or node in community2 or node in community3):
          if(node in graph):
            graph[node].append(neigh)
          else:
            graph[node] = [neigh]
        if(neigh in community1 or neigh in community2 or neigh in community3):
          if(neigh in graph):
            graph[neigh].append(node)
          else:
            graph[neigh] = [node]
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
label = np.zeros((len(graph), 3))
for nodeId in node_index:
  if(nodeId in community1):
    label[node_index[nodeId]] = [1, 0, 0]
  elif(nodeId in community2):
    label[node_index[nodeId]] = [0, 1, 0]
  else:
    label[node_index[nodeId]] = [0, 0, 1]
train_label, test_index = trainData(label)
feature_matrix = np.eye(len(adj_matrix))
adj_matrix = adj_matrix + feature_matrix
train_index = []
for i in range(len(adj_matrix)):
  if i not in test_index:
    train_index.append(i)
def gnn(adj_matrix, degree_matrix, label, feature_matrix, test_label, test_index, train_index):
  features = tf.placeholder(tf.float32, shape = ((None,len(feature_matrix))))
  adjacency = tf.placeholder(tf.float32, shape = ((None,None)))
  degree = tf.placeholder(tf.float32, shape = ((None,None)))
  labels = tf.placeholder(tf.float32, shape = ((None,label.shape[1])))
  weights1 = tf.Variable(tf.random_normal([len(feature_matrix),3], stddev = 1))
  #weights2 = tf.Variable(tf.random_normal([512, 3], stddev = 1))
  trainIndex = tf.placeholder(tf.int32, shape = ((len(train_index))))

  def layer(features, adjacency, degree, weights):
    with tf.name_scope('gcn_layer'):
      d_ = tf.pow(tf.matrix_inverse(degree), 0.5)
      y = tf.matmul(d_, tf.matmul(adjacency, d_))
      kernel = tf.matmul(features, weights)
      return tf.nn.relu(tf.matmul(y, kernel))
#     hidden1 = layer(features, adjacency, degree, weights1)
#     hidden1 = tf.layers.dropout(hidden1, rate=0.5)
    model = layer(features, adjacency, degree, weights1)
    training_output = tf.gather(model, trainIndex)
    #training_output = model[train_index]
    training_label = label[train_index]
    
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = training_output, labels = training_label))
      train_op = tf.train.AdamOptimizer(0.001, 0.9).minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    test_label = test_label[test_index]
    b = np.argmax(test_label, axis = 1)
    a = []
    for i in range(3501):
      _, cost = sess.run([train_op, loss], feed_dict = {features: feature_matrix, adjacency: adj_matrix, degree: degree_matrix, labels: label, trainIndex: train_index})
      if(i%100 == 0):
        print(cost)
        predict = sess.run(tf.nn.softmax(model), feed_dict = {features: feature_matrix, adjacency: adj_matrix, degree: degree_matrix, labels: test_label})
        test_res = predict[test_index]
        a = np.argmax(test_res, axis = 1)
        print("test accuracy: ", np.sum(a == b)/len(test_index))  
    return a 