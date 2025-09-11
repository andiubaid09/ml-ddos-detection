from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Prediksi di training set untuk mengetahui overfit sebuah model
y_train_pred = model.predict(X_train)

# Hitung akurasi
train_acc = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_acc)

# Evaluate model menggunakan datasheet test
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print('Akurasi :', acc)

# Grafik evaluasi confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

label = ['Normal','DDoS']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels = label,
            yticklabels = label)
plt.xlabel('Predicted Label')
plt.ylabel('True label')
plt.title('Confusion Matrix')
plt.show()

# Top Fitur XGBoost
from xgboost import plot_importance
plot_importance(model, importance_type='gain')
plt.show()
