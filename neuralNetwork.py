import numpy as np


#################################################################################
#########            Acitvation function and derivative Classes       ###########
#################################################################################

def relu(ZLinearCombination):
    return 1*ZLinearCombination

def sigmoid(ZLinearCombination):
    return 1*ZLinearCombination


#################################################################################
#########                  L-layer Neural Network Class               ###########
#################################################################################

class neuralNetwork:
    """ A class implementing a neural network with L layers.
    
    Attributes:
        architecture = a numpy ndarray (row vector) where each index denotes the layer and the element at the index indicating the number of neurons/nodes in that layer. The length of the ndarray indicates the number of layers in the neural network.
        
        activation = a string representing the activation function used for the hidden layers of the neural network.
        
    Methods:
        train(XData, yData, alpha, batchSize):
            A method passing in input XData and output yData training data for updating weights of the neural network using back-propogation. alpha is a float indicating the learning rate of backpropagation and batchSize is an int that indicates the number of training data to use per iteration of training.
            
        test(self, XData):
            Forward passing input data XData through the neurl network. Returns a output vector y with expected output.
                
    
    todo:
        - make the activation a row vector where each index denotes the layer and the respective element would be a string that represents the activation function that should be used at that layer.
    """
    def __init__(self,architecture,activationFunc = "relu"):
        #self.numLayers = numLayers
        self.architecture = architecture
        self.activationFunc = activationFunc
        self.Weights = None
        self.bais = None
    
    def train(self, XData, yData, alpha, batchSize):
        self.Weights = [None]                                   # initialized with no weights for input layer
        self.bais = [None]                                      # initialized with no weights for input layer

        for layer in range(1,self.architecture.shape[1]):     #initializing weights
            print(layer) ##########
            self.Weights.append(np.random.randn(self.architecture[0][layer],self.architecture[0][layer-1]))
            self.bais.append(np.zeros((self.architecture[0][layer],1)))
        
    def test(self, XData):
        if self.Weights is None:
            return "Error: NN not trained"
        
        self.ZLinearCombination = []                # calculates linear combination of inputs at each layer
        self.activation = []                        # calculated activation at each layer
        
        self.ZLinearCombination.append(XData)
        self.activation.append(XData)
        
        for layer in range(1,self.architecture.shape[1]):    #iterate through each layer calculating linear combination of inputs and activations.
            print(layer, "  ",self.architecture.shape[1]) #########
            print(len(self.ZLinearCombination),len(self.activation))
            Zcurr = np.dot(self.Weights[layer],self.activation[layer-1])+self.bais[layer]
            
            self.ZLinearCombination.append(Zcurr)
            if self.activationFunc == "relu":
                self.activation.append(relu(Zcurr))
            elif self.activationFunc == "sigmoid":
                self.activation.append(sigmoid(Zcurr))
            
            if layer == self.architecture.shape[1]:         # Edge case where output is a value between 0,1
                self.activation[layer] = sigmoid(Zcurr)
        
        return self.activation[self.architecture.shape[1]-1]
                
        
       
