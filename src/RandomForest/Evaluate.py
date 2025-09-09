from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns

#Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.tight_layout()
plt.show()
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
labels = ['Normal (0)','DDoS (1)']

print('=== Evaluasi Model ===')
print(f'Akurasi         : {accuracy:.4f}')
print(f'F1-Score Normal : {f1[0]:.4f}')
print(f'F1-Score DDoS   : {f1[1]:.4f}')
print(f'Recall Normal   : {recall[0]:.4f}')
print(f'Recall DDoS     : {recall[1]:.4f}')
print("\n=== Classifacation Report ===")
print(classification_report(y_test, y_pred, digits=4))

#Grafik F1-Score
x = np.arange(len(labels))
bar_width = 0.6
plt.figure(figsize=(6,4))
bars = plt.bar(x, f1, width=bar_width, color='Orange')
plt.xticks(x, labels)
plt.ylim(0,1.05)
plt.ylabel('F1-Score')
plt.title('F1-score')
for bar in bars:
  height = bar.get_height()
  plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0,3), textcoords='offset points', ha='center')
plt.tight_layout()
plt.show()

#Grafik Recall
plt.figure(figsize=(6,4))
bars = plt.bar(x, recall, width=bar_width, color='Purple')
plt.xticks(x,labels)
plt.ylim(0, 1.05)
plt.ylabel("Recall")
plt.title("Recall per Kelas")
for bar in bars:
  height = bar.get_height()
  plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
  xytext=(0,3), textcoords='offset points', ha='center')

plt.tight_layout()
plt.show()
importances = rf.feature_importances_
Features = x_train.columns

importance_df = pd.DataFrame({
    'Fitur': Features,
    'Importance': importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df.head(10))
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Fitur', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Fitur Paling Berpengaruh (Random Forest)')
plt.xlabel('Tingkat Kepentingan')
plt.ylabel('Fitur')
plt.tight_layout()
plt.show()
