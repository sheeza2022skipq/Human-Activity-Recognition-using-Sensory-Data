import numpy as np
from f_utils import *
import copy



    
def check_gradients(self, train_X, train_t):
        ## add code here            
                
        eps= 1e-5
        grad_ok = 0
        
        self.fprop(train_X)
        self.bprop(train_t)
        
        for l in range(1, self.num_layers+1):  
            ## add code here 
            Analytical_grad = self.grads["dW%s" % l]
            W = self.parameters["W%s" % l]
            Numerical_grad = np.zeros(W.shape)
    
            for i in range(0, W.shape[0]):
                for j in range(0, W.shape[1]):
                    W_one = copy.deepcopy(W)
                    W_one[i, j] = W_one[i, j] + eps
                    self.parameters["W%s" % l] = W_one
                    self.fprop(train_X)
                    loss_one = self.calculate_loss(train_t)
            
            
                    W_two = copy.deepcopy(W)
                    W_two[i, j] = W_two[i, j] - eps
                    self.parameters["W%s" % l] = W_two
                    self.fprop(train_X)
                    loss_two = self.calculate_loss(train_t)
            
                    Num_grad = (loss_one - loss_two)/(2*eps)
            
                    Numerical_grad[i, j] = Num_grad
            diff = (np.linalg.norm(Numerical_grad - Analytical_grad))  / (np.linalg.norm(Numerical_grad) + np.linalg.norm(Analytical_grad))
            # print(Numerical_grad, Analytical_grad)
                      
            if (diff> eps):
                print("layer %s gradients are not ok"% l)  
                grad_ok = 0
            else:
                print("layer %s gradients are ok"% l)
                grad_ok = 1
              
        return grad_ok
         

            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            