lr = LR("Batch", 0.001, 5000)
lr.fit(X_train, y_train, X_test, y_test, "Batch")