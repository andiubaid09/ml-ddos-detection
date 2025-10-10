best_model_lbgm = random_searchlbgm.best_estimator_
print(best_model_lbgm)

# Pengujian menggunakan dataset training
from sklearn.metrics import accuracy_score

y_pred_train = best_model_lbgm.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
print(f"Akurasi model dengan dataset training :{acc_train:.4f}")
y_pred_test = best_model_lbgm.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print(f"Akurasi model dengan dataset test : {acc_test:.4f}")

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, y_pred_test))

# Validasi akurasi dengan cross validation
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model_lbgm, X,y, cv=kf)
print(scores.mean())

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model_lbgm, X, y, cv=skf)
print("Mean CV Accuracy:", scores.mean())

# Visualisasi 
import matplotlib.pyplot as plt
import seaborn as sns

label = ['Normal','DDoS']
cm = confusion_matrix(y_test, y_pred_test)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels= label)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
model_lbgm = best_model_lbgm.named_steps['classifier'] # Ambil model lbgm
preprocessor_fit = best_model_lbgm.named_steps['preprocess'] # Ambil nama fitur dari pipeline
