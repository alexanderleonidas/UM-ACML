import numpy as np

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)

class MSE(Loss):
    def forward(self, y_pred, y_true):
        # performs mean square error
        return np.mean(np.power(y_pred-y_true, 2))

    def backward(self, y_pred, y_true):
        # derivative of mean square error
        return 2 * (y_pred - y_true) / len(y_pred)