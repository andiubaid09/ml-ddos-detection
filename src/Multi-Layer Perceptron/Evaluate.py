from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib as plt
import seaborn as sns

y_pred = np.argmax(model.predict(X_test_prep), axis = 1)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi model dengan data pengujian : {acc:.4f}")
print(classification_report(y_test, y_pred))
label = ['Normal','DDoS']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label,
            yticklabels=label)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()