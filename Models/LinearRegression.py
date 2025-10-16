import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import h5py
from sklearn.preprocessing import StandardScaler


class LR:
    def __init__(self, optimizer="Batch", learning_rate=0.01, iterations=50):
        self.params = {}
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.loss_test = []
        self.loss_train_mae = []
        self.loss_test_mae = []
        self.loss_train_rmse = []
        self.loss_test_rmse = []
        self.r2_test = []
        self.X = None
        self.y = None
                
    def init_weights(self):
        np.random.seed(1)
        self.params["W1"] = np.random.randn(self.X.shape[1], 1)
        self.params['b1']  = np.random.randn(1,1)

    def mse_loss(self, y, yhat):
        return np.mean((yhat - y) ** 2)
    
    def rmse_loss(self, y, yhat):
        return np.sqrt(np.mean((yhat - y) ** 2))
    
    def mae_loss(self, y, yhat):
        return np.mean(np.abs(yhat - y))
    
    def r2_score(self, y, yhat):
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def forward_propagation(self, optimizer="Batch", randomKey=0):
        if optimizer == "Batch":
            yhat = self.X.dot(self.params['W1']) + self.params['b1']
            loss = self.mse_loss(self.y, yhat)
        elif optimizer == "SGD":
            yhat = self.X[randomKey, :].dot(self.params['W1']) + self.params['b1']
            loss = self.mse_loss(self.y[randomKey, :], yhat)
        return yhat, loss

    def back_propagation(self, yhat, optimizer="Batch", randomKey=0):
        if optimizer == "Batch":
            gradient_wrt_b = np.sum(yhat - self.y) / self.X.shape[0]
            gradient_wrt_W = self.X.T.dot(yhat - self.y) / self.X.shape[0]
        elif optimizer == "SGD":
            gradient_wrt_b = np.sum(yhat - self.y[randomKey, :]) / randomKey.size
            gradient_wrt_W = self.X[randomKey, :].T.dot(yhat - self.y[randomKey, :]) / randomKey.size
            gradient_wrt_b = gradient_wrt_b.clip(min=-6.5, max=6.5)
            gradient_wrt_W = gradient_wrt_W.clip(min=-6.5, max=6.5)
        self.params['W1'] -= self.learning_rate * gradient_wrt_W
        self.params['b1'] -= self.learning_rate * gradient_wrt_b

    def fit(self, X, y, X_test, y_test, optimizer="Batch", batch_size=32):
        self.X = X
        self.y = y
        self.init_weights()
        
        for i in range(self.iterations):
            if optimizer == "SGD":
                for _ in range(self.X.shape[0] // batch_size):
                    randomKey = np.random.randint(0, self.X.shape[0], size=batch_size)
                    yhat, loss = self.forward_propagation("SGD", randomKey)
                    self.back_propagation(yhat, "SGD", randomKey)
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)

            self.loss.append(loss)
            preds_train = self.predict(self.X)
            self.loss_train_mae.append(self.mae_loss(self.y, preds_train))
            self.loss_train_rmse.append(self.rmse_loss(self.y, preds_train))
            
            preds_test = self.predict(X_test)
            self.loss_test.append(self.mse_loss(y_test, preds_test))
            self.loss_test_mae.append(self.mae_loss(y_test, preds_test))
            self.loss_test_rmse.append(self.rmse_loss(y_test, preds_test))
            self.r2_test.append(self.r2_score(y_test, preds_test))
            
            print(f"Epoch {i+1}: MSE Train {loss:.4f} - MSE Test {self.loss_test[-1]:.4f} "
                  f"- MAE Test {self.loss_test_mae[-1]:.4f} - RMSE Test {self.loss_test_rmse[-1]:.4f} "
                  f"- R2 Test {self.r2_test[-1]:.4f}")

    def predict(self, X):
        return X.dot(self.params['W1']) + self.params['b1']
    
    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return (self.rmse_loss(y_test, preds),
                self.mse_loss(y_test, preds),
                self.mae_loss(y_test, preds),
                self.r2_score(y_test, preds))

    def plot_loss(self, optimizer="Batch"):
        plt.subplots(1, 3, figsize=(18, 5))
        plt.suptitle(f"Using {optimizer} optimizer with learning rate {self.learning_rate}", fontsize=14)
        
        plt.subplot(1,3,1)
        plt.plot(self.loss, label='Train MSE')
        plt.plot(self.loss_test, label='Test MSE')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("MSE Loss")
        
        plt.subplot(1,3,2)
        plt.plot(self.loss_train_mae, label='Train MAE')
        plt.plot(self.loss_test_mae, label='Test MAE')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("MAE Loss")
        
        plt.subplot(1,3,3)
        plt.plot(self.loss_train_rmse, label='Train RMSE')
        plt.plot(self.loss_test_rmse, label='Test RMSE')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("RMSE Loss")
        plt.title("RMSE Loss")
        
        plt.show()




# --- Load and preprocess data ---
data = pd.read_csv("Data\SolarPrediction.csv")
X = data.drop(["UNIXTime","Radiation"],axis=1)
Y = pd.DataFrame(data.loc[:,"Radiation"])

X['TSR_Minute'] = pd.to_datetime(X['TimeSunRise']).dt.minute
X['TSS_Minute'] = pd.to_datetime(X['TimeSunSet']).dt.minute
X['TSS_Hour'] = np.where(pd.to_datetime(X['TimeSunSet']).dt.hour==18, 1, 0)
X['Month'] = pd.to_datetime(X['Data']).dt.month
X['Day'] = pd.to_datetime(X['Data']).dt.day
X['Hour'] = pd.to_datetime(X['Time']).dt.hour
X['Minute'] = pd.to_datetime(X['Time']).dt.minute
X['Second'] = pd.to_datetime(X['Time']).dt.second
X = X.drop(['Data', 'Time', 'TimeSunRise', 'TimeSunSet'], axis=1)
X['WindDirection(Degrees)_bin'] = np.digitize(X['WindDirection(Degrees)'], np.arange(0.0, 1.0, 0.02).tolist())
X['TSS_Minute_bin'] = np.digitize(X['TSS_Minute'], np.arange(0.0, 288.0, 12).tolist())
X['Humidity_bin'] = np.digitize(X['Humidity'], np.arange(32, 3192, 128).tolist())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

# --- Linear Regression Model ---
lr = LR("Batch", learning_rate=0.001, iterations=6000)
lr.fit(X_train, y_train, X_test, y_test, optimizer="Batch")
lr.plot_loss("Batch")

# Print final metrics for LR
rmse_lr, mse_lr, mae_lr, r2_lr = lr.evaluate(X_test, y_test)
print("\nLinear Regression Performance on Test Data:")
print(f"MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, MAE: {mae_lr:.4f}, R2: {r2_lr:.4f}")

# Save LR weights
with h5py.File("LR_Weights.h5", 'w') as f:
    f.create_dataset("W1", data=lr.params["W1"])
    f.create_dataset("b1", data=lr.params["b1"])

