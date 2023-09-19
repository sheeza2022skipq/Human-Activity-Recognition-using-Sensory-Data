import numpy as np
import matplotlib.pyplot as plt
from f_utils import *
import copy
from f_check_gradient import *
import sys
import sklearn.metrics 
import math

class NeuralNetwork():
  
    def __init__(self, args):     
        self.num_neurons = args.layer_dim
        self.activations_func = args.activations
        self.learning_rate = args.learning_rate
        self.num_iterations = args.epochs
        self.mini_batch_size = args.batch_size
        self.num_layers = len(self.num_neurons) - 1
        self.parameters = dict()
        self.net = dict()
        self.grads = dict()
        self.velocity = dict()
        self.scale = dict()
        self.optimizer = args.optimizer
        self.loss = args.loss

        
    def random_initialize_parameters(self):
        #random intialization
        print('random_initialize_parameters')
        np.random.seed(2)
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):   
            self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) 
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            print("layer", str(l),"weights and biases", self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)
            
    def random_initialize_parameters_with_constant10(self): 
        print('random_initialize_parameters_with_constant10')
        #random intialization method , c=10   
        np.random.seed(2)
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):   
            self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) * 10
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            print("layer", str(l),"weights and biases", self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)


    def xavier_intialization_parameters(self):
        #xavier intialization method
        print('xavier_intialization_parameters')
        np.random.seed(2)
        print("Num of layers", self.num_layers)
        for l in range(1, self.num_layers + 1):
            if self.activations_func[l] == 'relu': # xavier intialization method
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l-1]/2.)
            else:                
                self.parameters['W%s' % l] = np.random.randn(self.num_neurons[l], self.num_neurons[l-1]) / np.sqrt(self.num_neurons[l - 1])
            self.parameters['b%s' % l] = np.zeros((self.num_neurons[l], 1))
            print("layer", str(l),"weights and biases", self.parameters['W%s'%l].shape,self.parameters['b%s'%l].shape)
            
   
    def fprop(self, batch_input, train=True): 
        self.net['Z%s' % 0] = batch_input
        
        
        for l in range(1, self.num_layers + 1):
            
            activation = self.activations_func[l]
            self.net['A%s' % l] = np.matmul(self.parameters['W%s' % l], self.net['Z%s' % (l-1)]) + self.parameters['b%s' % l]
                
            if activation == "tanh":
                self.net['Z%s' % l] = tanh(self.net['A%s' % l])
            elif activation == "relu":
                self.net['Z%s' % l] = relu(self.net['A%s' % l])
            elif activation == "lrelu":
                self.net['Z%s' % l] = lrelu(self.net['A%s' % l],0.1)
            elif activation == "sigmoid":
                self.net['Z%s' % l] = sigmoid(self.net['A%s' % l])
            elif activation == "identity":
                self.net['Z%s' % l] = identity(self.net['A%s' % l])
            elif activation == "softmax":
                self.net['Z%s' % l] = softmax(self.net['A%s' % l])
                
       
        


            
    def calculate_loss(self, batch_target):   
        # add code here
        L = self.num_layers
        N = batch_target.shape[1]

        #loss = 1/N * (- np.sum(np.sum(np.multiply(batch_target , np.log(self.net['Z%s' %L] + 1e-8)),axis=1 )))
        loss = (1/N) * (-np.sum(np.multiply(batch_target,np.log(self.net['Z%s' %L] + 1e-8)) - np.multiply(1-batch_target, np.log(1-self.net['Z%s' %L]+ 1e-8))))
        return loss
        
        
    def update_parameters(self, epoch, t):
        # add code here
        #np.seterr(divide='ignore', invalid='ignore')
        if self.optimizer == 'sgd':
            
            for l in range(1,self.num_layers + 1):
                self.parameters['W%s' % l] = self.parameters['W%s' %l] - (self.learning_rate * self.grads['dW%s' %l])
                self.parameters['b%s' % l] = self.parameters['b%s' %l] - (self.learning_rate * self.grads['db%s' %l])
         
        elif self.optimizer == 'momentum':  
            coeff = 0.9
          
            for l in range(1,self.num_layers + 1):
                self.velocity['dW%s' % l] = np.zeros((self.grads["dW%s" % l].shape))  
                self.velocity['db%s' % l] = np.zeros((1,1))
                self.velocity['dW%s' % l] = (coeff * self.velocity['dW%s' % l]) - (self.learning_rate * self.grads["dW%s" % l])
                self.velocity['db%s' % l] = (coeff * self.velocity['db%s' % l]) - (self.learning_rate * self.grads["db%s" % l]) 
                self.parameters['W%s' % l] = self.parameters['W%s' %l] +  self.velocity['dW%s' % l]
                self.parameters['b%s' % l] = self.parameters['b%s' %l] +  self.velocity['db%s' % l]
            
        elif self.optimizer == 'adam':
            
            eps = 1e-8
            scaling_decay  = 0.95
            momentum_decay = 0.9
            for l in range(1,self.num_layers + 1):
                #m
                self.velocity['dW%s' % l] = np.zeros((self.grads["dW%s" % l].shape))  
                self.velocity['db%s' % l] = np.zeros((1,1))
                self.scale['dW%s' % l] = np.zeros((self.grads["dW%s" % l].shape))  
                self.scale['db%s' % l] = np.zeros((1,1))
                self.velocity['dW%s' % l] = momentum_decay * self.velocity['dW%s' % l] + (1- momentum_decay ) * self.grads["dW%s" % l]
                self.velocity['db%s' % l] = momentum_decay * self.velocity['db%s' % l] + (1- momentum_decay ) * self.grads["db%s" % l]
                self.velocity['dW%s' % l] = np.divide(self.velocity['dW%s' % l] , (1 - momentum_decay**t) +eps) 
                self.velocity['db%s' % l] = np.divide(self.velocity['db%s' % l] , (1 - momentum_decay**t) +eps) 
                #v
                self.scale['dW%s' % l] = scaling_decay * self.scale['dW%s' % l] + (1- scaling_decay) * (self.grads["dW%s" % l]**2)
                self.scale['db%s' % l] = scaling_decay * self.scale['db%s' % l]+ (1- scaling_decay ) * (self.grads["db%s" % l]**2)
                self.scale['dW%s' % l] = np.divide(self.scale['dW%s' % l] , (1 - scaling_decay**t) +eps)
                self.scale['db%s' % l] = np.divide(self.scale['db%s' % l] , (1 - scaling_decay**t)+eps )
                
                self.parameters['W%s' % l] = self.parameters['W%s' %l] - np.divide((self.learning_rate * self.velocity['dW%s' % l]) , (np.sqrt(self.scale['dW%s' % l] + eps)))
                self.parameters['b%s' % l] = self.parameters['b%s' %l] - np.divide((self.learning_rate * self.velocity['db%s' % l]) , (math.sqrt(self.scale['db%s' % l]) + eps))
                t+=1
            
    
        
     
    def bprop(self, batch_target):
        # add code here 
        L = self.num_layers

        m = batch_target.shape[1]
        self.net['dZ%s'%L] = np.subtract(self.net['Z%s'%L] , batch_target)
        self.grads['dW%s' %L] =  1/m * np.matmul( self.net['dZ%s'%L], self.net['Z%s'%(L-1)].T )
        self.grads['db%s' %L] =  1/m * np.sum(np.array(self.net['dZ%s'%L]))
        
        for l in range(self.num_layers-1 , 0,-1):
            mul = np.matmul(self.parameters['W%s' %(l+1)].T ,self.net['dZ%s'%(l+1)])
            if self.activations_func[l] == "tanh":
                derivate = tanh_derivative(self.net['A%s'%l])
                
            elif self.activations_func[l] == "relu":
                derivate = relu_derivative(self.net['A%s'%l])
                
            elif self.activations_func[l] == "sigmoid":
                derivate = sigmoid_derivative(self.net['A%s'%l])
            
            elif self.activations_func[l] == "lrelu":
                derivate = lrelu_derivative(self.net['A%s'%l],0.1)


            self.net['dZ%s'%l] = mul * derivate

            self.grads['dW%s' %l] =  1/m * np.dot( self.net['dZ%s'%l] , self.net['Z%s'%(l-1)].T )
            self.grads['db%s' %l] =  1/m * np.sum(np.array(self.net['dZ%s'%l]))
   
           

    def plot_loss(self,loss,val_loss):        
        plt.figure()
        fig = plt.gcf()
        plt.plot(loss, linewidth=3, label="train")
        plt.plot(val_loss, linewidth=3, label="val")
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('learning rate =%s, hidden layers=%s' % (self.learning_rate, self.num_layers-1))
        plt.grid()
        plt.legend()
        plt.show()
        fig.savefig('plot_loss.png')
        
    
    def plot_gradients(self):
        avg_l_g = []
        grad = copy.deepcopy(self.grads)
        for l in range(1, self.num_layers+1):
             weights_grad = grad['dW%s' % l]  
             dim = weights_grad.shape[0]
             avg_g = []
             for d in range(dim):
                 abs_g = np.abs(weights_grad[d])
                 avg_g.append(np.mean(abs_g))             
             temp = np.mean(avg_g)
             avg_l_g.append(temp)   
        layers = ['layer %s'%l for l in range(self.num_layers+1)]
        weights_grad_mag = avg_l_g
        fig = plt.gcf()
        plt.xticks(range(len(layers)), layers)
        plt.xlabel('layers')
        plt.ylabel('average gradients magnitude')
        plt.title('')
        plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2) 
        plt.show() 
        fig.savefig('plot_gradients.png')
    

    def train(self, train_x, train_y, val_x, val_y):  
        #self.random_initialize_parameters()      
        #self.random_initialize_parameters_with_constant10()
        self.xavier_intialization_parameters()
        train_loss = []
        val_loss = []
        num_samples = train_y.shape[1]       
        check_grad = True
        grad_ok = 0


        for i in range(0, self.num_iterations):
            for idx in range(0, num_samples, self.mini_batch_size):
                minibatch_input =  train_x[:, idx:idx + self.mini_batch_size]
                minibatch_target =  train_y[:, idx:idx + self.mini_batch_size]
#                 if check_grad == True:
#                     grad_ok = check_gradients(self, minibatch_input, minibatch_target)               
                    
#                     if grad_ok == 0:                           
#                         print("gradients are not ok!\n")                           
#                         sys.exit()
                grad_ok=1
                if grad_ok == 1:
                    check_grad = False
                    self.fprop(minibatch_input)
                    loss = self.calculate_loss(minibatch_target)
                    self.bprop(minibatch_target)           
                    self.update_parameters(i,0)
                    
            train_loss.append(loss)     
            self.fprop(val_x)
            va_loss = self.calculate_loss(val_y)
            val_loss.append(va_loss) 
            print("Epoch %i: training loss %f, validation loss %f" % (i, loss,va_loss))
            
        self.plot_loss(train_loss,val_loss)     
        self.plot_gradients()      
        
        
        
    def test(self, test_x, test_y):
       #test_x, test_y = shuffle(test_x, test_y, random_state=0)
        target = np.argmax(test_y, axis = 0)
        self.fprop(test_x)
        
        output = np.argmax(self.net['Z%s' % self.num_layers], axis = 0)
        print(output)
        print(target)
        equl = output == target

        accuracy = np.mean(equl) * 100
        
        return accuracy, output
