import math
import random

import numpy as np

class Neuron:
    def __init__(self, edgeWeights=[], inputValue=0, error=0, output=0):
        self.edgeWeights = edgeWeights #used to store weights of edges connected to a unit in a neural network
        self.inputValue = inputValue #used to store input taken by a neural network
        self.error = error #represents error for a unit in a neural network
        self.output = output #represents output for a unit in a neural network
 
class NeuralNetwork:
    
    def __init__(self, numInputUnits, numHiddenUnits, numOutputUnits, lowerLimitForRandNums, upperLimitForRandNums, learningRate):
        
        '''
        numInputUnits = number of input units in a neural network
        numHiddenUnits = number of hidden units in a neural network
        numOutputUnits = number of output units in a neural network
        lowerLimitForRandNums = minimum value for a random number
        upperLimitForRandNums = maximum value for a random number
        learningRate = a float representing a learning rate
        
        NOTE: all weights for a neural network will be initialized to a number >= lowerLimitForRandNums and <= upperLimitForRandNums 
        '''
        
        #represents output units for neural network
        self.outputUnits = [Neuron() for x in range(numOutputUnits)]
        
        #represents hidden units for neural network
        self.hiddenUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numOutputUnits)]) for x in range(numHiddenUnits)]
        
        #represents input units for neural network
        self.inputUnits = [Neuron([random.uniform(lowerLimitForRandNums, upperLimitForRandNums) for y in range(numHiddenUnits)]) for x in range(numInputUnits)]
        
        #represents learning rate for neural network
        self.learningRate = learningRate

    # THESE TWO FUNCTIONS MIGHT NOT WORK
    def fit(self, X, y):
        pass
    # THESE TWO FUNCTIONS MIGHT NOT WORK
    def predict(self, X):
        y = []
        for x in X:
            self.doForwardPropagation(x)
            y.append(np.array([u.output for u in self.outputUnits]))
        
        return y
    
    #*********************************************************************************************************#
    
    #this function is used when back propagation is being done
    #updates the error for all the output units in a neural network
    def updateOutputUnitErrors(self, targetValues):
        
        '''
        targetValues = list of values that will be compared with the outputs of 
        all the output units in a neural network
        '''
        
        '''
        for each output unit, use the formula \delta_k = (o_k) * (1 - o_k) * (t_k - o_k) to 
        compute the most up to date output
        '''
        for x in range(0, len(self.outputUnits)):
            output = self.outputUnits[x].output
            target = targetValues[x]
            self.outputUnits[x].error = (output) * (1 - output) * (target - output) 
        #

    #
        
    #*********************************************************************************************************#

    #this function is used when back propagation is being done
    #updates the error for all the hidden units in a neural network
    def updateHiddenUnitErrors(self):

        #go through every hidden unit in a neural network
        for x in range(0, len(self.hiddenUnits)):
            
            hiddenUnit = self.hiddenUnits[x]
            edgeWeights = hiddenUnit.edgeWeights
            
            #use output unit errors to help you calculate the error of a hidden unit
            numOutputUnits = len(self.outputUnits)
            outputUnitErrors = [self.outputUnits[h].error for h in range(numOutputUnits)]
            total = 0

            for y in range(0, len(edgeWeights)):
                product = edgeWeights[y] * outputUnitErrors[y]
                total += product
            #
            
            #calculate error of a hidden unit
            output = self.hiddenUnits[x].output
            self.hiddenUnits[x].error = output * (1 - output) * total
            
        #   

    #
    
    #*********************************************************************************************************#
    
    #updates all the weights for a neural network
    def updateNetworkWeights(self):
        
        #go through every input unit in a neural network
        for x in range(0, len(self.inputUnits)):
            
            #represents an input unit
            inputUnit = self.inputUnits[x]
            
            #represents value that an input unit transfers to a hidden unit
            inputValue = inputUnit.inputValue
            
            #update the weights of edges connecting an input unit to all hidden units in a neural network
            for y in range(0, len(self.hiddenUnits)): 
                
                #represents error of a hidden unit
                delta = self.hiddenUnits[y].error 
                
                #update weight of edge connecting an input unit to a hidden unit
                self.inputUnits[x].edgeWeights[y] = self.inputUnits[x].edgeWeights[y] + (delta * inputValue * self.learningRate)
                
            #
            
        #
        
        #go through every hidden unit in a neural network
        for x in range(0, len(self.hiddenUnits)):
            
            #represents a hidden unit
            hiddenUnit = self.hiddenUnits[x]
            
            #represents value that a hidden unit transfers to an output unit
            inputValue = hiddenUnit.inputValue
            
            #update the weights of edges connecting a hidden unit to all output units in a neural network
            for y in range(0, len(self.outputUnits)): 
                
                #represents error of output unit
                delta = self.outputUnits[y].error 

                #update weight of edge connecting a hidden unit to an output unit
                self.hiddenUnits[x].edgeWeights[y] = self.hiddenUnits[x].edgeWeights[y] + (delta * inputValue * self.learningRate)
                
            #
            
        #
        
    #
    
    #*********************************************************************************************************#
    
    #used to perform back propagation
    def doBackwardPropagation(self, targetValues):
        
        '''
        targetValues = list of values that will be compared with the outputs of 
        all the output units in a neural network
        '''
        
        #update output unit errors
        NeuralNetwork.updateOutputUnitErrors(self, targetValues.copy()) 
        
        #update hidden unit errors
        NeuralNetwork.updateHiddenUnitErrors(self)
        
        #update weights for a neural network
        NeuralNetwork.updateNetworkWeights(self)
        
    #
    
    #*********************************************************************************************************#
    
    #this function is used when forward propagation is being done
    #updates the output for a hidden unit
    def updateOutputForHiddenUnit(self, index):
        
        '''
        index = a number >= 0, used to refer to some hidden unit
        '''
        
        #represents up to date output for a hidden unit
        output = 0
        
        '''
        To update the output for a hidden unit ...
        - use the weights of edges connecting all the input units to a hidden unit
        - use the values that all the input units will transfer to a hidden unit during forward propagation
        '''
        for x in range(0, len(self.inputUnits)): 
            
            #represents an input unit
            inputUnit = self.inputUnits[x]
            
            #represents value that an input unit will transfer to a hidden unit
            inputValue = inputUnit.inputValue
            
            #represents weight of edge connecting an input unit to a hidden unit
            weight = inputUnit.edgeWeights[index]
            
            #represents product of the two variables above
            product = inputValue * weight 

            output += product 
        #
        
        self.hiddenUnits[index].output = NeuralNetwork.sigmoid(self, output)
           
    #
    
    #*********************************************************************************************************#
    
    #this function is used when forward propagation is being done
    #updates the output for an output unit
    def updateOutputForOutputUnit(self, index):
        
        '''
        index = a number >= 0, used to refer to some output unit
        '''
        
        #represents up to date output for an output unit
        output = 0
        
        '''
        To update the output for an output unit ...
        - use the weights of edges connecting all the hidden units to an output unit
        - use the values that all the hidden units will transfer to an output unit during forward propagation
        '''
        for x in range(0, len(self.hiddenUnits)): 
            
            #represents a hidden unit
            hiddenUnit = self.hiddenUnits[x] 
            
            #represents value that a hidden unit will transfer to an output unit
            inputValue = hiddenUnit.inputValue 
            
            #represents weight of edge connecting a hidden unit to an output unit
            weight = hiddenUnit.edgeWeights[index] 
            
            #represents product of the two variables above
            product = inputValue * weight 
            
            output += product 
        #
        
        self.outputUnits[index].output = NeuralNetwork.sigmoid(self, output) 
        
    #
    
    #*********************************************************************************************************#
    
    #serves as a sigmoid function
    def sigmoid(self, outputValue):
        
        '''
        input: outputValue = output of a unit in a neural network
        
        output: a decimal between 0 and 1
        '''
        
        negatedOutput = -1 * outputValue
        denominator = 1 + math.pow(math.e, negatedOutput)
        result = (1.0) / (1.0 * denominator)
            
        return result
               
    #
    
    #*********************************************************************************************************#
   
    #used to perform forward propagation
    def doForwardPropagation(self, inputVector):
        
        '''
        inputVector = list of values that will be fed to the input units
        '''
        
        #store input values in input units
        for x in range(0, len(self.inputUnits)): 
            
            #use input units to store values present in input vector
            self.inputUnits[x].inputValue = inputVector[x] 
            
        #
        
        #compute outputs for all hidden units
        for x in range(0, len(self.hiddenUnits)):
            
            #compute the output for a hidden unit
            NeuralNetwork.updateOutputForHiddenUnit(self, x)  
            
            #a hidden unit will transfer its output to an output unit
            self.hiddenUnits[x].inputValue = self.hiddenUnits[x].output
            
        #
        
        #compute outputs for all output units
        for x in range(0, len(self.outputUnits)):
            
            #compute the output for an output unit
            NeuralNetwork.updateOutputForOutputUnit(self, x) 
            
        #
        
    #    
    
    #*********************************************************************************************************#
    
    #used to help a neural network learn from a training example
    def trainOnExample(self, inputVector, outputVector):
        
        '''
        input: inputVector = list of numbers that will be fed to the input units, 
        outputVector = list of values that will be compared with the outputs of all output units
        
        output: a list of the format [hiddenUnitOutputs, outputUnitOutputs], where hiddenUnitOutputs is a
        list of outputs for all hidden units and outputUnitOutputs is a list of outputs for all output units
        '''
        
        #perform forward propagation
        NeuralNetwork.doForwardPropagation(self, inputVector.copy()) 
        
        #create the list that we want to return from this function
        numHiddenUnits = len(self.hiddenUnits) 
        numOutputUnits = len(self.outputUnits) 
        hiddenUnitOutputs = [self.hiddenUnits[x].output for x in range(numHiddenUnits)] 
        outputUnitOutputs = [self.outputUnits[x].output for x in range(numOutputUnits)]
        array = [hiddenUnitOutputs, outputUnitOutputs]
        
        #perform backward propagation
        NeuralNetwork.doBackwardPropagation(self, outputVector.copy())
        
        #clear all garbage values that are found in a neural network
        itemsToClean = [self.inputUnits, self.hiddenUnits, self.outputUnits] 
        
        for h in range(0, len(itemsToClean)): 
            item = itemsToClean[h] 
            for x in range(0, len(item)): 
                item[x].error = 0 
                item[x].output = 0 
                item[x].inputValue = 0
            #
        #
        
        return array
        
    #
    
    #*********************************************************************************************************#
    
#

def main():

    network = NeuralNetwork(numInputUnits=3, numHiddenUnits=2, numOutputUnits=1, lowerLimitForRandNums=-.05, upperLimitForRandNums=.05, learningRate=.1)

    X = np.array([
        [-1, 0, 0],
        [-1, 0, 1],
        [-1, 1, 0],
        [-1, 1, 1],
    ])
    target_and = np.array([0, 0, 0, 1])
    target_or = np.array([0, 1, 1, 1])
    target_xor = np.array([0, 1, 1, 0])
    target_nand = np.array([1, 1, 1, 0])
    target_nor = np.array([1, 0, 0, 0])

    targets = [
        (target_and, "AND"), 
        (target_or, "OR"), 
        (target_xor, "XOR"), 
        (target_nand, "NAND"), 
        (target_nor, "NOR")
    ]
   
    epochs = 1000
    for y, name in targets:
        print(f"Training {name} for {epochs} epochs")
        for _ in range(0, epochs):
            for x, t in zip(X, [y]):
                network.trainOnExample(x, t)
        print("Predicting for AND")
        y_hat = network.predict(X)
        print(f"\tExpected output: {y}")
        print(f"Actual Output: {y_hat}")

if __name__ == "__main__":
    main()