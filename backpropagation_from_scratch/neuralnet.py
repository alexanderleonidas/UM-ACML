from layer import Layer
from activation import Activation
from const import *

class NerualNet:
    def __init__(self):
        self.network = [Layer(INPUT_SIZE,HIDDEN_SIZE), Activation(), Layer(HIDDEN_SIZE,OUTPUT_SIZE), Activation()]
        self.loss_history = []
    
    def train(self, x, y, loss_func, learning_rate=0.1, epochs=100000):
        """Training function for the neural network"""
        
        for epoch in range(epochs):
            # Forward pass
            output = x
            for layer in self.network:
                layer.forward(output)
                output = layer.output
            
            # Calculate loss
            loss = loss_func.calculate(output, y)
            self.loss_history.append(loss)
            
            if epoch % 10000 == 0:
                print(f'epoch: {epoch}, loss: {loss}')
            
            # Backward pass
            loss_grad = loss_func.backward(output, y)
            
            dvalues = loss_grad
            for layer in reversed(self.network):
                layer.backward(dvalues)
                dvalues = layer.dinputs
            
            # Update weights and biases
            for layer in self.network:
                if isinstance(layer, Layer):
                    layer.weights -= learning_rate * layer.dweights
                    layer.bias -= learning_rate * layer.dbias
        
        return loss