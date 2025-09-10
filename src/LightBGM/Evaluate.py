#Prediksi Model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score

y_predik = model.predict(X_test)
akurasi = accuracy_score(y_test, y_predik)
print(f"Akurasi : {akurasi:.4f}")
print(classification_report(y_test, y_predik))
#Membuat grafik evaluasi confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

label = ['Normal','DDoS']
cm = confusion_matrix(y_test, y_predik)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels = label,
            yticklabels = label)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

import numpy as np

precision = precision_score(y_test, y_predik, average=None)
recall = recall_score(y_test, y_predik, average=None)
f1 = f1_score(y_test, y_predik, average=None)
classes = ['Normal','DDoS']

X = np.arange(len(classes))
width = 0.2

plt.bar(X - width, precision, width, label='Precision')
plt.bar(X, recall, width, label ='Recall')
plt.bar(X + width, f1, width, label='F1-score')
plt.xticks(X, classes)
plt.ylabel('Score')
plt.ylim(0,1.1)
plt.title('Metrics per Class')
plt.legend()
plt.show

import lightgbm as lgb

lgb.plot_importance(model)
plt.title("Top Feature Importance")
plt.show()
