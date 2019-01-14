import numpy as np
import tensorflow as tf
from utils import *

def gnn(adj_matrix, degree_matrix, label, feature_matrix, test_label, test_index, train_index):
    features = tf.placeholder(tf.float32, shape = ((None,len(feature_matrix))))
    adjacency = tf.placeholder(tf.float32, shape = ((None,None)))
    degree = tf.placeholder(tf.float32, shape = ((None,None)))
    labels = tf.placeholder(tf.float32, shape = ((None,3)))
    weights1 = tf.Variable(tf.random_normal([len(feature_matrix),512], stddev = 1))
    weights2 = tf.Variable(tf.random_normal([512, 3], stddev = 1))
    trainIndex = tf.placeholder(tf.int32, shape = ((len(train_index))))

    def layer(features, adjacency, degree, weights):
        with tf.name_scope('gcn_layer'):
            d_ = tf.pow(tf.matrix_inverse(degree), 0.5)
            y = tf.matmul(d_, tf.matmul(adjacency, d_))
            kernel = tf.matmul(features, weights)
            return tf.nn.relu(tf.matmul(y, kernel))
    hidden1 = layer(features, adjacency, degree, weights1)
    hidden1 = tf.layers.dropout(hidden1, rate=0.5)
    model = layer(hidden1, adjacency, degree, weights2)
    training_output = tf.gather(model, trainIndex)
    #training_output = model[train_index]
    training_label = label[train_index]
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = training_output, labels = training_label))
        train_op = tf.train.AdamOptimizer(0.01, 0.9).minimize(loss)
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    test_label = test_label[test_index]
    b = np.argmax(test_label, axis = 1)
    for i in range(2000):
        _, cost = sess.run([train_op, loss], feed_dict = {features: feature_matrix, adjacency: adj_matrix, degree: degree_matrix, labels: label, trainIndex: train_index})
        if(i%100 == 0):
            print(cost)
            predict = sess.run(tf.nn.softmax(model), feed_dict = {features: feature_matrix, adjacency: adj_matrix, degree: degree_matrix, labels: test_label})

            test_res = predict[test_index]
            a = np.argmax(test_res, axis = 1)
            print("test accuracy: ", np.sum(a == b)/len(test_index))


if __name__ == '__main__':
  community1, community2, graph = loadData('com-youtube.top5000.cmty.txt', 'com-youtube.ungraph.txt')
  adj_matrix, degree_matrix, label = preprocessingData(graph, community1, community2)
  train_label, test_index = trainData(label)
  feature_matrix = np.eye(len(adj_matrix))
  adj_matrix = adj_matrix + feature_matrix
  train_index = []
  for i in range(len(adj_matrix)):
    if i not in test_index:
      train_index.append(i)
  gnn(adj_matrix, degree_matrix, train_label, feature_matrix, label, test_index, train_index)
  

    