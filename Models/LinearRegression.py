import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle

class LR:
    def __init__(self, optimizer="Batch", learning_rate=0.01, iterations=50):
        np.random.seed(1)  # Set seed once during initialization
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.params = {}
        self.loss = []
        self.loss_test = []
        self.loss_train_mae = []
        self.loss_test_mae = []
        self.loss_train_rmse = []
        self.loss_test_rmse = []
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def init_weights(self):
        """Initialize weights using normal distribution."""
        self.params["W1"] = np.random.randn(self.X.shape[1], 1)
        self.params["b1"] = np.random.randn(1, 1)

    def mse_loss(self, y, yhat):
        return np.mean(np.power(yhat - y, 2))

    def rmse_loss(self, y, yhat):
        return np.sqrt(self.mse_loss(y, yhat))

    def mae_loss(self, y, yhat):
        return np.mean(np.abs(yhat - y))

    def forward_propagation(self):
        """Performs forward propagation."""
        yhat = self.X.dot(self.params["W1"]) + self.params["b1"]
        return yhat, self.mse_loss(self.y, yhat)

    def back_propagation(self, yhat):
        """Computes gradients and updates weights."""
        gradient_wrt_b = np.sum(yhat - self.y) / self.X.shape[0]
        gradient_wrt_W = np.dot(self.X.T, yhat - self.y) / self.X.shape[0]

        self.params["W1"] -= self.learning_rate * gradient_wrt_W
        self.params["b1"] -= self.learning_rate * gradient_wrt_b

    def fit(self, X_train, y_train, X_test, y_test, optimizer_type="Batch"):
        """Trains the model using the specified optimizer."""
        self.optimizer = optimizer_type  # Store the optimizer type
        self.X = self.scaler.fit_transform(X_train)
        self.y = y_train
        self.init_weights()

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

            preds_train = self.predict(X_train)
            loss_train_mae = self.mae_loss(y_train, preds_train)
            loss_train_rmse = self.rmse_loss(y_train, preds_train)
            self.loss_train_mae.append(loss_train_mae)
            self.loss_train_rmse.append(loss_train_rmse)

            preds = self.predict(X_test)
            loss_test = self.mse_loss(y_test, preds)
            loss_test_mae = self.mae_loss(y_test, preds)
            loss_test_rmse = self.rmse_loss(y_test, preds)

            self.loss_test_mae.append(loss_test_mae)
            self.loss_test_rmse.append(loss_test_rmse)
            self.loss_test.append(loss_test)

            print(f"Epoch {i+1}: MSE Training Loss {loss:.4f} - MSE Testing Loss {loss_test:.4f} - "
                  f"MAE Testing Loss {loss_test_mae:.4f} - RMSE Testing Loss {loss_test_rmse:.4f}")

        # Save optimal weights
        with open("optimal_weights.pkl", "wb") as f:
            pickle.dump(self.params, f)

    def predict(self, X):
        """Predicts using the trained model."""
        X_scaled = self.scaler.transform(X)
        return np.round(X_scaled.dot(self.params["W1"]) + self.params["b1"])

    def load_weights(self, filename="optimal_weights.pkl"):
        """Loads trained weights from file."""
        with open(filename, "rb") as f:
            self.params = pickle.load(f)

    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data."""
        preds = self.predict(X_test)
        return self.rmse_loss(y_test, preds), self.mse_loss(y_test, preds), self.mae_loss(y_test, preds)

    def plot_loss(self):
        """Plots loss curves."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(self.loss, label="Training MSE Loss")
        plt.plot(self.loss_test, label="Testing MSE Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("MSE Loss Curve")

        plt.subplot(1, 3, 2)
        plt.plot(self.loss_train_mae, label="Training MAE Loss")
        plt.plot(self.loss_test_mae, label="Testing MAE Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.title("MAE Loss Curve")

        plt.subplot(1, 3, 3)
        plt.plot(self.loss_train_rmse, label="Training RMSE Loss")
        plt.plot(self.loss_test_rmse, label="Testing RMSE Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("RMSE Loss")
        plt.title("RMSE Loss Curve")

        plt.tight_layout()
        plt.show()
