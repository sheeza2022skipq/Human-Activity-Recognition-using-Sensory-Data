"""
@author: arbish

"""
import numpy as np
from neural_network import *
from f_load_data import *
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('--layer_dim', dest='layer_dim', type=list, default=[1440, 50, 50, 2], help='no of layers with neurons')
parser.add_argument('--activations', dest='activations', type=list, default=[None, 'lrelu','lrelu', 'sigmoid'], help='activation function')
parser.add_argument('--optimizer', dest='optimizer', default='adam', help='sgd, momentum, adam')
parser.add_argument('--epochs', dest='epochs', type=int, default=10, help='epochs')
parser.add_argument('--loss', dest='loss', default='cross-entropy', help='mse, cross-entropy')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=100, help='batch_size')
parser.add_argument('--learning_rate', dest='learning_rate', type=int, default=1e-3, help='learning_rate')

args = parser.parse_args()


nn = NeuralNetwork(args)

# load dataset
train_x, train_t, val_x, val_t, test_x, test_t = load_dataset()
print("train_x and train_t: ", train_x.shape, train_t.shape)
print("val_x and val_t: ", val_x.shape, val_t.shape)
print("test_x and test_t: ", test_x.shape, test_t.shape)


nn.train(train_x, train_t, val_x, val_t)
train_acc,confusion_matrix = nn.test(train_x, train_t)
test_acc, confusion_matrix = nn.test(test_x, test_t) 
print("training acc..", np.round(train_acc,4))
print("testing acc..", np.round(test_acc,4))


#[784, 10, 20, 30, 50, 100, 10]
#[784, 50, 10]

#[None, 'tanh','tanh','relu','lrelu','tanh','softmax']
#[None, 'tanh', 'softmax']
