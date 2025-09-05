# Hyperparameter yang digunakan 
from pprint import pprint

print('Hyperparameter yang sedang digunakan:')
pprint(model_rf.get_params())

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

#Confusion Matrix
cm = confusion_matrix(y_test, predict)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.tight_layout()
plt.show()

#Evaluasi Model RandomForest
accuracy = accuracy_score(y_test, predict)
f1 = f1_score(y_test, predict, average=None)
recall = recall_score(y_test, predict, average=None)
labels = ['Normal (0)','DDoS (1)']

print('=== Evaluasi Model ===')
print(f'Akurasi         : {accuracy:.4f}')
print(f'F1-Score Normal : {f1[0]:.4f}')
print(f'F1-Score DDoS   : {f1[1]:.4f}')
print(f'Recall Normal   : {recall[0]:.4f}')
print(f'Recall DDoS     : {recall[1]:.4f}')
print("\n=== Classifacation Report ===")
print(classification_report(y_test, predict, digits=4))

importances = model_rf.feature_importances_
Features = X_train.columns

importance_df = pd.DataFrame({
    'Fitur': Features,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Fitur', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Fitur Paling Berpengaruh (Random Forest)')
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()
