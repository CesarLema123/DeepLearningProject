import numpy as np
import matplotlib.pyplot as plt

#################################################################################
#########             Acitvation function and derivatives             ###########
#################################################################################

def relu(ZLinearCombination):
    """ ReLU Function implementation
    Arguements:
        ZLinearCombination = a numpy ndarray
    Returns:
        ReLU(ZLinearCombination)
    """
    np.maximum(0,ZLinearCombination)
    return ZLinearCombination
    
def reluPrime(ZLinearCombination):
    """ ReLU derivative Function implementation
    Arguements:
        ZLinearCombination = a numpy ndarray
    Returns:
        ReLU'(ZLinearCombination)
    """
    relu(ZLinearCombination)                                          # Getting rid of negative values
    ZLinearCombination[ZLinearCombination > 0] = 1                    # Returning slope for non-negative values
    return ZLinearCombination
    
def sigmoid(ZLinearCombination):
    """ Sigmoid Function implementation
    Arguements:
        ZLinearCombination = a numpy ndarray
    Returns:
        sigmoid(ZLinearCombination)
    """
    return (1+np.exp(-1*ZLinearCombination))**(-1)
    
def sigmoidPrime(ZLinearCombination):
    """ Sigmoid derivative Function implementation
    Arguements:
        ZLinearCombination = a numpy ndarray
    Returns:
        sigmoid'(ZLinearCombination)
    """
    return sigmoid(ZLinearCombination)*(1-sigmoid(ZLinearCombination))


#################################################################################
#########                  L-layer Neural Network Class               ###########
#################################################################################

class neuralNetwork:
    """ A class implementing a neural network with L layers and arbitrary number of nodes. Hidden Layers can be specified to have either ReLU or sigmoid activation functions while the output layer is implemented with a sigmoid activation function. The empirical risk or cost is caculated using logistic loss.
    
    Attributes:
        architecture = a numpy ndarray (row vector) where each index denotes the layer and the element at the index indicating the number of neurons/nodes in that layer. The length of the ndarray indicates the number of layers in the neural network.
        
        activationFunc = a string representing the activation function used for the hidden layers of the neural network.
        
    Methods:
        train(XData, yData, alpha, iterations, batchSize):
            takes in input XData and output yData training data for updating weights of the neural network using back-propogation. alpha is a float indicating the learning rate of backpropagation, iterations is an int indicating the number of iterations of backpropogation and batchSize is an int that indicates the number of training data to use per iteration of training.
            
        test(self, XData):
            Forward passing of input data XData through the neurl network. Returns a output vector y with expected/calculated output.
                
        empiricalRisk(self, yData,output):
            computes the ER / cost of output values calculated by NN.
        
        train_dLdA, train_dLdZ, train_dRdW, train_dRdb are private methods used to implement the train method. They are implemented with the formalism for partial derivatives discussed in notes.
    
    todo:
        - Implement batchSize in train parameter
        
        - Make the activation a row vector where each index denotes the layer and the respective element would be a string that represents the activation function that should be used at that layer.
        
        - Allow output and hidden layer activation functions to be initializable.
        
        - Fix problems with calculating nan values
    """
    def __init__(self, architecture, activationFunc = "relu"):
        self.architecture = architecture
        self.activationFunc = activationFunc
        self.numLayers = self.architecture.shape[1]-1             # Convention: input= 0 layer, output= L Layer
        
        self.Weights = [None]                       # Initialized with no weights for input layer
        self.bais = [None]                          # Initialized with no weights for input layer

        for layer in range(1,self.numLayers+1):       # Initializing weights
            self.Weights.append(np.random.randn(self.architecture[0][layer],self.architecture[0][layer-1]))
            self.bais.append(np.zeros((self.architecture[0][layer],1)))
    
    def train(self, XData, yData, alpha, iterations, batchSize):
        self.numSamples = XData.shape[1]
        self.test(XData)                                  # calculate Z and activation values needed for backprop
        self.empRisk = []
        self.empRisk.append(self.empiricalRisk(yData,self.activation[self.numLayers]))  # Record Cost
        
        for iter in range( iterations-1):                   # For specified number of iterations
            dLdA_Curr = self.train_dLdA(self.numLayers, None, yData)          # calc for output layer
            dLdZ_Curr = self.train_dLdZ(self.numLayers, dLdA_Curr)
            dLdZ_prev = dLdZ_Curr
            dRdW_Curr = self.train_dRdW(self.numLayers,dLdZ_Curr)
            dRdb_Curr = self.train_dRdb(self.numLayers, dLdZ_Curr )
            self.Weights[self.numLayers] = self.Weights[self.numLayers] - (alpha * dRdW_Curr)                 # Updating weights
            self.bais[self.numLayers] = self.bais[self.numLayers] - (alpha * dRdb_Curr)                       # Updating bais

            for layer in range(self.numLayers-1,0,-1):    # Iterate backwards through each layer calculating derivatives for updates and applying them.
                dLdA_Curr = self.train_dLdA(layer, dLdZ_prev, yData)          # calc for hidden layers
                dLdZ_Curr = self.train_dLdZ(layer, dLdA_Curr)
                dLdZ_prev = dLdZ_Curr
                dRdW_Curr = self.train_dRdW(layer,dLdZ_Curr)
                dRdb_Curr = self.train_dRdb(layer, dLdZ_Curr)
                self.Weights[layer] = self.Weights[layer] - (alpha*dRdW_Curr)          # Updating weights
                self.bais[layer] = self.bais[layer] - (alpha*dRdb_Curr)                # Updating bais
            
            self.test(XData)                                  # recalculate Z and activation values needed for backprop after updates
            self.empRisk.append(self.empiricalRisk(yData,self.activation[self.numLayers])) # Record Cost
                
    
    def test(self, XData):
        self.ZLinearCombination = []                # Linear combination of inputs at each layer
        self.activation = []                        # Activation at each layer
        
        self.ZLinearCombination.append(XData)
        self.activation.append(XData)
        
        for layer in range(1,self.numLayers+1):    # Iterate through each layer calculating linear combination of inputs and activation values.
            Zcurr = np.dot(self.Weights[layer],self.activation[layer-1])+self.bais[layer]
            
            self.ZLinearCombination.append(Zcurr)
            if self.activationFunc == "relu":               # Apply specified activation function
                self.activation.append(relu(Zcurr))
            elif self.activationFunc == "sigmoid":
                self.activation.append(sigmoid(Zcurr))
            
            if layer == self.numLayers:                     # Edge case: output with sigmoid activation
                self.activation[layer] = sigmoid(Zcurr)

        return self.activation[self.numLayers]
        
    def empiricalRisk(self, yData,output):
        loss = ((yData - 1)*(np.log(1 - output))) - (yData*np.log(output))  # Logistic loss
        return np.mean(loss, axis=1 , keepdims = True)
    
    def train_dLdA(self, layer, dLdz_Prev, yData = None):
        if layer == self.numLayers:                          # Calculation of initial derivative value
            return ((yData-1)*(-1*(1-self.activation[self.numLayers])**(-1))) - (yData*(self.activation[self.numLayers]**(-1)))
        else:                                               # Calculation of derivative for any other layer
            return np.dot(self.Weights[layer+1].T, dLdz_Prev)
    
    def train_dLdZ(self, layer , dLdA_Curr):
        if layer == self.numLayers or self.activationFunc == "sigmoid":        # Calculation at output layer where activation is sigmoid or a hidden layer with sigmoid activation
            return dLdA_Curr * sigmoidPrime(self.ZLinearCombination[layer])
        elif self.activationFunc == "relu":                                    # Calculation at hidden layer with relu activation function
            return dLdA_Curr * reluPrime(self.ZLinearCombination[layer])
        #elif self.activationFunc == "tanh":                                   # todo
    
    def train_dRdW(self, layer, dldZ_Curr):
        return (self.numSamples**(-1))* np.dot(dldZ_Curr,self.activation[layer-1].T)
        
    def train_dRdb(sellf, layer, dldZ_Curr):                  # this is a batch calculation for bais update
        return np.mean(dldZ_Curr,axis = 1,keepdims=True)

#################################################################################
#########                          Test Code                          ###########
#################################################################################

architecture = np.array([2,2,5,1]).reshape(1,4)
iterations = 100
XData = np.array([[1,3,5,3],[1,3,5,3]]).reshape(2,4)
#XData = np.random.randn(2,4)
#XData = np.array([[10,30,5,3],[1,3,5,3]]).reshape(2,4)         # Problems with large input values often returning nan during some calculations
yData = np.array([[0.4]])

model = neuralNetwork(architecture)
model.train(XData,yData,0.001,iterations,32)

output = model.test(XData)
print("Output: ",output)
#print("Z: ",model.ZLinearCombination)
#print("Number of Layers: ", model.numLayers )

plt.plot(range(0,iterations),np.ravel(model.empRisk))               # Plotting Cost / ER from training
plt.xlabel("Iteration")
plt.ylabel("Empirical Risk")
plt.title("Empirical Risk vs. Iteration")
plt.grid()
plt.show()
