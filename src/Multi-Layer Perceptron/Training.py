history = model.fit(
    X_train_prep, y_train,
    validation_split = 0.2,
    epochs = 50,
    validation_data = (X_val_prep, y_val),
    verbose = 1
)
test_loss, test_acc = model.evaluate(X_test_prep, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f} | Test acc :{test_acc:.4f}")