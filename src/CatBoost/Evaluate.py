best_modelCAT = random_search.best_estimator_
print(best_modelCAT)

# Pengujian menggunakan dataset training

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred_train = best_modelCAT.predict(X_train)
acc = accuracy_score(y_train, y_pred_train)
print(f"Akurasi model dengan data training : {acc:.4f}")

# Pengujian menggunakan dataset uji

y_predi_test = best_modelCAT.predict(X_test)
acc_test = accuracy_score(y_test, y_predi_test)
print(f"Akurasi model dengan data pengujian : {acc_test:.4f}")
print(classification_report(y_test,y_predi_test))
