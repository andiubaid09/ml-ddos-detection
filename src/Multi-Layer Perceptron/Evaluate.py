from sklearn.metrics import accuracy_score,classification_report
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi: {akurasi:.4f}")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics import confusion_matrix, f1_score

labels = ['Normal', 'DDoS']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# F1 Score
f1 = f1_score(y_test, y_pred, average=None)

plt.figure(figsize=(6,4))
bars = plt.bar(labels, f1, color=['skyblue', 'salmon'])
plt.title('F1 Score per Class')
plt.ylim(0,1.05)
plt.ylabel('F1 Score')

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0,3), textcoords='offset points', ha='center')

plt.show()
