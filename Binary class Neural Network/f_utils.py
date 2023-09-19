import numpy as np

def normalize_data(data):   
    eps=1e-8
    mean = np.mean(data, axis=1, keepdims=True)
    variance = np.var(data, axis=1, keepdims=True)
    data_norm = np.divide((data - mean), np.sqrt(variance+eps))
    return data_norm

def tanh(a):
    return np.tanh(a)

def tanh_derivative(a):
    A = tanh(a)
    der = 1 - np.square(A)
    return der

def relu(a):
    return np.maximum(0,a)
  
def sigmoid(a):
    return 1/(1 + np.exp(-a))

def relu_derivative(a): 
    re = np.zeros(a.shape, dtype = np.int64())
    
    re[a<=0] = 0
    re[a>0] = 1
    
    return re

def sigmoid_derivative(a):
    der = sigmoid(a) * ( 1 - sigmoid(a))
    return der

def lrelu(a, k):
    lre = np.zeros(a.shape)
    
    lre[a<=0] = k * a[a<=0]
    lre[a>0] = a[a>0]

    return lre

def lrelu_derivative(a, k):  
    der = np.zeros(a.shape)
    
    der[a<=0] = k
    der[a>0] = 1

    return der


def identity(a):
    return a

def softmax(a):
    max_a = np.max(a, axis=0, keepdims=True)
    a_exp = np.exp(a - max_a)
    a_sum = np.sum(a_exp, axis=0, keepdims=True)
    softmax = np.divide(a_exp, a_sum)
    return softmax


