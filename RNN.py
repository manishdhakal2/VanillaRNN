

import numpy as np

class RecurrentNetwork:
    """Simple Recurrent Network
        Expects One-Hot Encoded Vectors as an Input"""
    
    #------------------------------------------------------------------------------------------------#
    

    def __init__(self,lr,neurons):
        self.lr=lr
        self.neurons=neurons


    #------------------------------------------------------------------------------------------------#

    def init_weights(self,index):
        """"Initializes the weights and biases"""
        self.weightW=np.random.uniform(0,1,size=(self.neurons)) #Initialize random weights in the range [0,1) #Weight map of the input
        self.weightU=np.random.uniform(0,1,size=(self.neurons)) #Initialize random weights in the range [0,1) #Weight map of the hidden state
        self.bias=0

    #------------------------------------------------------------------------------------------------#

    def __ReLU___(self,x):
        """Returns the ReLU activated output of any input"""
        return np.max(0,x) 

    
    #------------------------------------------------------------------------------------------------#

    def forward(self,x:np.array):
        """Returns the output of the forward propagation
            *Input Shape=(batch_size,sequence_size,one_hot_length)"""
        
        for i in range(x.shape[1]): #Input (batch_size,sequence_size,...) 
            #Loop over all the elements in the sequence
            out=np.dot(self.x,self.weightW)+np.dot(self.hidden_state,self.weightU)+self.bias
            activatedOut=self.__ReLU___(out)
            self.hidden_state=activatedOut
        return activatedOut
    
        #------------------------------------------------------------------------------------------------#


    def backward(self):
        """Performs a backward propagation updating all the weights and biases of the model at once"""
        pass
