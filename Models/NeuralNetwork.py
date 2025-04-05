import numpy as np
import matplotlib.pyplot as plt

class Net:
   class Net:
    import numpy as np
import matplotlib.pyplot as plt

class Net:
    def __init__(self, layers=[16, 64, 48, 32, 1], learning_rate=0.01, iterations=50, lambda_reg=0.01, patience=10):
       
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.loss = []
        self.loss_test = []
        self.loss_train_mae = []
        self.loss_test_mae = []
        self.loss_train_rmse = []
        self.loss_test_rmse = []
        self.layers = layers
        self.X = None
        self.y = None
        self.best_val_loss = float('inf')
        self.wait = 0


    def evaluate(self, X_test, y_test):  # Added the evaluate method
        """Evaluates the model on test data."""
        preds = self.predict(X_test)
        return self.rmse_loss(y_test, preds), self.mse_loss(y_test, preds), self.mae_loss(y_test, preds)

    

    def fit(self, X, y,X_test,y_test,optimizer="Batch",batch_size=32):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()
        
        if(optimizer == "SGD"):
            for i in range(self.iterations):
                if((i+1)%50 == 0):
                    self.learning_rate/=2
                for _ in range(self.X.shape[0]//batch_size):
                    randomKey = np.random.randint(0,self.X.shape[0],size=batch_size)
                    yhat, loss = self.forward_propagation("SGD",randomKey)
                    self.back_propagation(yhat,"SGD",randomKey)
                yhat, loss = self.forward_propagation()
                self.back_propagation(yhat)
                self.loss.append(loss)
                preds_train = np.array(self.predict(self.X))
                true_outputs_train = np.array(self.y)
                loss_train_mae = self.mae_loss(true_outputs_train,preds_train)
                loss_train_rmse = self.rmse_loss(true_outputs_train,preds_train)
                self.loss_train_mae.append(loss_train_mae)
                self.loss_train_rmse.append(loss_train_rmse)
                preds = np.array(self.predict(X_test))
                true_outputs = np.array(y_test)
                loss_test = self.mse_loss(true_outputs,preds)
                loss_test_mae = self.mae_loss(true_outputs,preds)
                loss_test_rmse = self.rmse_loss(true_outputs,preds)
                self.loss_test_mae.append(loss_test_mae)
                self.loss_test_rmse.append(loss_test_rmse)
                self.loss_test.append(loss_test)
                print("Epoch {}: MSE Training Loss {} - MSE Testing Loss {} - MAE Testing Loss {} - RMSE Testing Loss {}".format(i+1, loss,loss_test,loss_test_mae,loss_test_rmse))
        else:
            for i in range(self.iterations):
                if((i+1)%100==0):
                    if((i+1) <= 400):
                        self.learning_rate /=2
                yhat, loss = self.forward_propagation()
                self.back_propagation(yhat)
                self.loss.append(loss)
                preds_train = np.array(self.predict(self.X))
                true_outputs_train = np.array(self.y)
                loss_train_mae = self.mae_loss(true_outputs_train,preds_train)
                loss_train_rmse = self.rmse_loss(true_outputs_train,preds_train)
                self.loss_train_mae.append(loss_train_mae)
                self.loss_train_rmse.append(loss_train_rmse)
                preds = np.array(self.predict(X_test))
                true_outputs = np.array(y_test)
                loss_test = self.mse_loss(true_outputs,preds)
                loss_test_mae = self.mae_loss(true_outputs,preds)
                loss_test_rmse = self.rmse_loss(true_outputs,preds)
                self.loss_test_mae.append(loss_test_mae)
                self.loss_test_rmse.append(loss_test_rmse)
                self.loss_test.append(loss_test)
                print("Epoch {}: MSE Training Loss {} - MSE Testing Loss {} - MAE Testing Loss {} - RMSE Testing Loss {}".format(i+1, loss,loss_test,loss_test_mae,loss_test_rmse))

    def init_weights(self):
        np.random.seed(1)
        # Ensure weight dimensions match the layers architecture
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) * np.sqrt(2/self.layers[0])
        self.params['b1'] = np.zeros((self.layers[1], 1))
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2]) * np.sqrt(2/self.layers[1])
        self.params['b2'] = np.zeros((self.layers[2], 1))
        self.params['W3'] = np.random.randn(self.layers[2], self.layers[3]) * np.sqrt(2/self.layers[2])
        self.params['b3'] = np.zeros((self.layers[3], 1))
        self.params['W4'] = np.random.randn(self.layers[3], self.layers[4]) * np.sqrt(2/self.layers[3])
        self.params['b4'] = np.zeros((self.layers[4], 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def dRelu(self, x):
        return np.where(x > 0, 1, 0)

    def mse_loss(self, y, yhat):
        return np.mean(np.power(yhat - y, 2))
    
    def rmse_loss(self, y, yhat):
        return np.sqrt(np.mean(np.power(yhat - y, 2)))
    
    def mae_loss(self, y, yhat):
        return np.mean(np.abs(yhat - y))

    def forward_propagation(self, optimizer="Batch", random_indices=None):
        if optimizer == "Batch":
            X = self.X
            y = self.y.reshape(-1, 1)
        else:  # Mini-Batch SGD
            X = self.X[random_indices] if random_indices is not None else self.X
            y = self.y[random_indices].reshape(-1, 1)
        
        Z1 = X.dot(self.params['W1']) + self.params['b1'].T
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2'].T
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3'].T
        A3 = self.relu(Z3)
        Z4 = A3.dot(self.params['W4']) + self.params['b4'].T
        yhat = Z4  # No ReLU on last layer

        loss = self.mse_loss(y, yhat) + self.lambda_reg * (
            np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])) +
            np.sum(np.square(self.params['W3'])) + np.sum(np.square(self.params['W4']))
        )
        
        self.params.update({'Z1': Z1, 'Z2': Z2, 'Z3': Z3, 'Z4': Z4, 'A1': A1, 'A2': A2, 'A3': A3})
        return yhat, loss

     # Backpropagation method with gradient clipping
    def back_propagation(self, yhat, optimizer="SGD", random_indices=None):
        # Calculate the error/loss gradient for the last layer (output layer)
        m = yhat.shape[0]  # Batch size
        y = self.y[random_indices] if random_indices is not None else self.y.reshape(-1, 1)
        X = self.X[random_indices] if random_indices is not None else self.X 

        # Output layer gradient
        dZ4 = yhat - y  # Derivative of the loss w.r.t. output Z4
        dW4 = (1/m) * self.params['A3'].T.dot(dZ4) + (self.lambda_reg / m) * self.params['W4']
        db4 = (1/m) * np.sum(dZ4, axis=0, keepdims=True)

        # Backpropagate the error through the 3rd layer
        dA3 = dZ4.dot(self.params['W4'].T)
        dZ3 = dA3 * self.dRelu(self.params['Z3'])  # Elementwise multiplication with the derivative of ReLU
        dW3 = (1/m) * self.params['A2'].T.dot(dZ3) + (self.lambda_reg / m) * self.params['W3']
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

        # Backpropagate the error through the 2nd layer
        dA2 = dZ3.dot(self.params['W3'].T)
        dZ2 = dA2 * self.dRelu(self.params['Z2'])
        dW2 = (1/m) * self.params['A1'].T.dot(dZ2) + (self.lambda_reg / m) * self.params['W2']
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)


        # Backpropagate the error through the 1st layer
        dA1 = dZ2.dot(self.params['W2'].T)
        dZ1 = dA1 * self.dRelu(self.params['Z1'])
        dW1 = (1/m) * X.T.dot(dZ1) + (self.lambda_reg / m) * self.params['W1']
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Gradient Clipping (Add this to avoid large gradients)
        grad_clip_value = 10  # Clip gradients to prevent explosion
        dW1 = np.clip(dW1, -grad_clip_value, grad_clip_value)
        dW2 = np.clip(dW2, -grad_clip_value, grad_clip_value)
        dW3 = np.clip(dW3, -grad_clip_value, grad_clip_value)
        dW4 = np.clip(dW4, -grad_clip_value, grad_clip_value)

        # Update the weights and biases using gradient descent (SGD)
        self.params['W1'] -= self.learning_rate * dW1
        self.params['b1'] -= self.learning_rate * db1.T
        self.params['W2'] -= self.learning_rate * dW2
        self.params['b2'] -= self.learning_rate * db2.T
        self.params['W3'] -= self.learning_rate * dW3
        self.params['b3'] -= self.learning_rate * db3.T
        self.params['W4'] -= self.learning_rate * dW4

    def predict(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1'].T
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2'].T
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3'].T
        A3 = self.relu(Z3)
        Z4 = A3.dot(self.params['W4']) + self.params['b4'].T
        return Z4  # No ReLU in the output layer

    def plot_loss(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(self.loss, label='Training Loss')
        plt.plot(self.loss_test, label='Testing Loss')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training & Testing MSE Loss")

        plt.subplot(1, 3, 2)
        plt.plot(self.loss_train_mae, label='Training MAE')
        plt.plot(self.loss_test_mae, label='Testing MAE')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("Training & Testing MAE Loss")

        plt.subplot(1, 3, 3)
        plt.plot(self.loss_train_rmse, label='Training RMSE')
        plt.plot(self.loss_test_rmse, label='Testing RMSE')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("RMSE Loss")
        plt.title("Training & Testing RMSE Loss")

        plt.show()
