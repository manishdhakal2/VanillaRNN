

import numpy as np

class RecurrentNetwork:
    """Simple Recurrent Network
        Expects One-Hot Encoded Vectors as an Input"""
    
    #------------------------------------------------------------------------------------------------#
    

    def __init__(self,lr,neurons):
        self.lr=lr
        self.neurons=neurons
        self.train_mode=False


    #------------------------------------------------------------------------------------------------#

    def init_weights(self,index):
        """"Initializes the weights and biases"""
        self.weightW=np.random.uniform(0,1,size=(self.neurons)) #Initializes random weights in the range [0,1) #Weight map of the input
        self.weightU=np.random.uniform(0,1,size=(self.neurons)) #Initializes random weights in the range [0,1) #Weight map of the hidden state
        
        self.bias=0

    #------------------------------------------------------------------------------------------------#

    def __ReLU___(self,x):
        """Returns the ReLU activated output of any input"""
        return np.max(0,x) 

    
    #------------------------------------------------------------------------------------------------#

    def forward(self,x:np.array):
        """Returns the output of the forward propagation
            *Input Shape=(batch_size,sequence_size,one_hot_length)"""
        
        self.hidden_state=None #Hidden State Is None for first element in the sequence
        
        

        self.hidden_states=np.zeros(x.shape[1])


        #Loops over all the elements in the sequence
        for i in range(x.shape[1]): #Input (batch_size,sequence_size,...) 

            #Computes the Forward Step
            if(i==0):
                #Skip the computation as self.hidden_state is None
                out=np.dot(x[i],self.weightW)+self.bias
            else:

                out=np.dot(x[i],self.weightW)+np.dot(self.hidden_state,self.weightU)+self.bias 
            

            #Applies Activation
            hidden_state=self.__ReLU___(out) 


            #Stores all instances of hidden state 
            self.hidden_state[i]=hidden_state
        return hidden_state
    
    #------------------------------------------------------------------------------------------------#

    def partial_forward(self,x,h):
        """Returns the forward pass for just one sequence
        parameters:
            x->Input vector
            h->hidden state"""
    
        return np.dot(x,self.weightW)+np.dot(h,self.weightU)+self.bias
        
    #------------------------------------------------------------------------------------------------#

    def backward(self):
        """Performs a backward propagation updating all the weights and biases of the model at once
        Currently lets assume the loss is MSE"""
        for i in self.hidden_state[::-1]:
            
            out=self.partial_forward()
    
    #------------------------------------------------------------------------------------------------#

    def train(self,x,y,epochs):

        """Trains The Simple RNN Model
        
        Params:
            x:Input Vectors
            y:Corresponding Labels/target
            epochs:No of epochs to train the model 
            """
        self.train_mode=True
        for i in range(epochs):
            self.forward(x)
            self.backward(x,y)

