import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

import h5py
np.random.seed(42)
class Net():
        
    def __init__(self, layers=[16,64,48,32,1], learning_rate=0.01, iterations=50):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.loss_test = []
        self.loss_train_mae = []
        self.loss_test_mae = []
        self.loss_train_rmse = []
        self.loss_test_rmse = []
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) * np.sqrt(2/self.layers[0])
        self.params['b1']  =np.random.randn(self.layers[1],)* np.sqrt(2/self.layers[0])
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) * np.sqrt(2/self.layers[1])
        self.params['b2'] = np.random.randn(self.layers[2],) * np.sqrt(2/self.layers[1])
        self.params["W3"] = np.random.randn(self.layers[2], self.layers[3]) * np.sqrt(2/self.layers[2])
        self.params['b3']  =np.random.randn(self.layers[3],)* np.sqrt(2/self.layers[2])
        self.params['W4'] = np.random.randn(self.layers[3],self.layers[4]) * np.sqrt(2/self.layers[3])
        self.params['b4'] = np.random.randn(self.layers[4],)* np.sqrt(2/self.layers[3])
    
    def relu(self,Z):
        '''
        The ReLu activation function is to performs a threshold
        operation to each input element where values less 
        than zero are set to zero.
        '''
        return np.maximum(0,Z)

    def dRelu(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def mse_loss(self,y, yhat):

        loss = np.mean(np.power(yhat-y, 2));
        return loss
    def rmse_loss(self,y, yhat):
    
        loss = np.sqrt(np.mean(np.power(yhat-y, 2)))
        return loss
    def mae_loss(self,y, yhat):
        
        loss = np.mean(np.abs(yhat-y))
        return loss
    def r2_score(self,y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)

    def forward_propagation(self,optimizer="Batch",randomKey=0):
        if(optimizer == "Batch"):
            Z1 = self.X.dot(self.params['W1']) + self.params['b1']
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            A2 = self.relu(Z2)
            Z3 = A2.dot(self.params['W3']) + self.params['b3']
            A3 = self.relu(Z3)
            Z4 = A3.dot(self.params['W4']) + self.params['b4']
            yhat = self.relu(Z4)
            loss = self.mse_loss(self.y,yhat)
        elif(optimizer == "SGD"):
            Z1 = self.X[randomKey,:].dot(self.params['W1']) + self.params['b1']
            A1 = self.relu(Z1)
            Z2 = A1.dot(self.params['W2']) + self.params['b2']
            A2 = self.relu(Z2)
            Z3 = A2.dot(self.params['W3']) + self.params['b3']
            A3 = self.relu(Z3)
            Z4 = A3.dot(self.params['W4']) + self.params['b4']
            yhat = self.relu(Z4)
            loss = self.mse_loss(self.y[randomKey,:],yhat)
    
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['Z3'] = Z3
        self.params['Z4'] = Z4
        self.params['A1'] = A1
        self.params['A2'] = A2
        self.params['A3'] = A3

        return yhat,loss

    def back_propagation(self,yhat,optimizer="Batch",randomKey=0):
        '''
        Computes the derivatives and update weights and bias according.
        '''
        if(optimizer == "Batch"):
            dl_wrt_yhat = (yhat-self.y)
            dl_wrt_z4 = dl_wrt_yhat * self.dRelu(self.params["Z4"])
            dl_wrt_w4 = self.params['A3'].T.dot(dl_wrt_z4)
            dl_wrt_b4 = np.sum(dl_wrt_z4, axis=0, keepdims=True)

            dl_wrt_A3 = dl_wrt_z4.dot(self.params['W4'].T)
            dl_wrt_z3 = dl_wrt_A3 * self.dRelu(self.params['Z3'])
            dl_wrt_w3 = self.params['A2'].T.dot(dl_wrt_z3)
            dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
            
            dl_wrt_A2 = dl_wrt_z3.dot(self.params['W3'].T)
            dl_wrt_z2 = dl_wrt_A2 * self.dRelu(self.params['Z2'])
            dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
            dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
            
            dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
            dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
            dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
            dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

            dl_wrt_w1 = dl_wrt_w1.clip(min=-6.5,max=6.5)
            dl_wrt_w2 = dl_wrt_w2.clip(min=-6.5,max=6.5)
            dl_wrt_w3 = dl_wrt_w3.clip(min=-6.5,max=6.5)
            dl_wrt_w4 = dl_wrt_w4.clip(min=-6.5,max=6.5)
            dl_wrt_b1 = dl_wrt_b1.clip(min=-6.5,max=6.5)
            dl_wrt_b2 = dl_wrt_b2.clip(min=-6.5,max=6.5)
            dl_wrt_b3 = dl_wrt_b3.clip(min=-6.5,max=6.5)
            dl_wrt_b4 = dl_wrt_b4.clip(min=-6.5,max=6.5)
            
        elif(optimizer == "SGD"):
            
            dl_wrt_yhat = (yhat-self.y[randomKey,:])
            dl_wrt_z4 = dl_wrt_yhat * self.dRelu(self.params["Z4"])
            dl_wrt_w4 = self.params['A3'].T.dot(dl_wrt_z4)
            dl_wrt_b4 = np.sum(dl_wrt_z4, axis=0, keepdims=True)

            dl_wrt_A3 = dl_wrt_z4.dot(self.params['W4'].T)
            dl_wrt_z3 = dl_wrt_A3 * self.dRelu(self.params['Z3'])
            dl_wrt_w3 = self.params['A2'].T.dot(dl_wrt_z3)
            dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
            
            dl_wrt_A2 = dl_wrt_z3.dot(self.params['W3'].T)
            dl_wrt_z2 = dl_wrt_A2 * self.dRelu(self.params['Z2'])
            dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
            dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
            
            dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
            dl_wrt_z1 = dl_wrt_A1 * self.dRelu(self.params['Z1'])
            dl_wrt_w1 = self.X[randomKey,:].T.dot(dl_wrt_z1)
            dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)

            dl_wrt_w1 = dl_wrt_w1.clip(min=-6.5,max=6.5)
            dl_wrt_w2 = dl_wrt_w2.clip(min=-6.5,max=6.5)
            dl_wrt_w3 = dl_wrt_w3.clip(min=-6.5,max=6.5)
            dl_wrt_w4 = dl_wrt_w4.clip(min=-6.5,max=6.5)
            dl_wrt_b1 = dl_wrt_b1.clip(min=-6.5,max=6.5)
            dl_wrt_b2 = dl_wrt_b2.clip(min=-6.5,max=6.5)
            dl_wrt_b3 = dl_wrt_b3.clip(min=-6.5,max=6.5)
            dl_wrt_b4 = dl_wrt_b4.clip(min=-6.5,max=6.5)
            
                
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['W3'] = self.params['W3'] - self.learning_rate * dl_wrt_w3
        self.params['W4'] = self.params['W4'] - self.learning_rate * dl_wrt_w4
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
        self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3
        self.params['b4'] = self.params['b4'] - self.learning_rate * dl_wrt_b4

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

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        A3 = self.relu(Z3)
        Z4 = A3.dot(self.params['W4']) + self.params['b4']
        pred = self.relu(Z4) 
        return pred

    
    def evaluate(self, x_test, y_test):
     preds = np.array(self.predict(x_test))
     true_outputs = np.array(y_test)
     metrics = {
        "RMSE": self.rmse_loss(true_outputs, preds),
        "MSE": self.mse_loss(true_outputs, preds),
        "MAE": self.mae_loss(true_outputs, preds),
        "R2": self.r2_score(true_outputs, preds)
     }
     return metrics

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.subplots(1,3)
        plt.subplot(1,3,1)
        plt.plot(self.loss,label='Training Loss')
        plt.plot(self.loss_test,label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Testing MSE Loss")
        plt.subplot(1,3,2)
        plt.plot(self.loss_train_mae,label='Training Loss')
        plt.plot(self.loss_test_mae,label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("Training and Testing MAE Loss")
        plt.subplot(1,3,3)
        plt.plot(self.loss_train_rmse,label='Training Loss')
        plt.plot(self.loss_test_rmse,label='Testing Loss')
        plt.legend(loc='best')
        plt.xlabel("Epoch")
        plt.ylabel("RMSE Loss")
        plt.title("Training and Testing RMSE Loss")
        plt.show()    
        

if __name__ == "__main__":
    # =================== Load and preprocess data ===================
    data = pd.read_csv(r"Data\SolarPrediction.csv")

    X = data.drop(["UNIXTime", "Radiation"], axis=1)
    Y = pd.DataFrame(data["Radiation"])  # Keep original units

    X['TSR_Minute'] = pd.to_datetime(X['TimeSunRise'], errors='coerce').dt.minute
    X['TSS_Minute'] = pd.to_datetime(X['TimeSunSet'], errors='coerce').dt.minute
    X['TSS_Hour'] = np.where(pd.to_datetime(X['TimeSunSet'], errors='coerce').dt.hour == 18, 1, 0)
    X['Month'] = pd.to_datetime(X['Data'], errors='coerce').dt.month
    X['Day'] = pd.to_datetime(X['Data'], errors='coerce').dt.day
    X['Hour'] = pd.to_datetime(X['Time'], format='%H:%M:%S', errors='coerce').dt.hour
    X['Minute'] = pd.to_datetime(X['Time'], format='%H:%M:%S', errors='coerce').dt.minute
    X['Second'] = pd.to_datetime(X['Time'], format='%H:%M:%S', errors='coerce').dt.second

    X = X.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)

    X['WindDirection(Degrees)_bin'] = np.digitize(
        X['WindDirection(Degrees)'], np.linspace(0, 360, 19)
    )
    X['TSS_Minute_bin'] = np.digitize(X['TSS_Minute'], np.arange(0, 288 + 12, 12))
    X['Humidity_bin'] = np.digitize(X['Humidity'], np.arange(32, 3200, 128))

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    with open("scaler_X.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    print("\nInput scaler saved to scaler_X.pkl")

    input_dim = X_train.shape[1]
    net = Net(layers=[input_dim, 64, 48, 32, 1], iterations=200, learning_rate=0.001)
    net.fit(X_train, np.asarray(y_train), X_test, np.asarray(y_test), optimizer="SGD", batch_size=64)

    metrics = net.evaluate(X_test, np.asarray(y_test))
    print("\nNeural Network Performance on Test Data:")
    print(f"MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, R2: {metrics['R2']:.4f}")

    net.plot_loss()

    with h5py.File("NN_Weights.h5", 'w') as f:
        f.create_dataset("W1", data=net.params["W1"])
        f.create_dataset("W2", data=net.params["W2"])
        f.create_dataset("W3", data=net.params["W3"])
        f.create_dataset("W4", data=net.params["W4"])
        f.create_dataset("b1", data=net.params["b1"])
        f.create_dataset("b2", data=net.params["b2"])
        f.create_dataset("b3", data=net.params["b3"])
        f.create_dataset("b4", data=net.params["b4"])
    print("\nWeights saved to NN_Weights.h5")
