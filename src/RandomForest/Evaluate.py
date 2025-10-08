best_model = random_search.best_estimator_
print(best_model)

# Melakukan pengetasan performa model menggunakan data training agar diliat overfit atau tidaknya sebuah model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_train_pred = best_model.predict(X_train)
acc = accuracy_score(y_train,y_train_pred)
print(f"Training Accuracy:{acc:.4f}")

# Melakukan pengujian performa menggunakan data pengujian
y_pred = best_model.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)
print(f'Akurasi model dari data pengujian adalah: {acc_test:.4f}')
print(classification_report(y_test,y_pred))

# Korelasi fitur dengan label
# Jika korelasi lebih dari >0.95 pada fitur, maka terjadi leakage data (kebocoran data)
korelasi = df[Features + ['label']].select_dtypes(include=['number']).corr()['label'].sort_values(ascending=False)
print(korelasi)

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
plt.title('Confusion Matrix')
plt.xlabel('Aktual')
plt.ylabel('Prediksi')
plt.tight_layout()
plt.show()

#Evaluasi Model RandomForest
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

preprocessor  = best_model.named_steps['preprocess']
model = best_model.named_steps['classifier']

# Ambil nama fitur hasil transformasi OneHotEncoder & StandarScaler
feature_names = preprocessor.get_feature_names_out()

# Ambil Feature Importances dari Model (RandomForest)
importances = model.feature_importances_

print(f"Jumlah feature names: {len(feature_names)}")
print(f"Jumlah Importances : {len(importances)}")

# Gabungkan ke DataFrame
feat_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Tampilkan top 10 feature paling penting
print("\nTop 10 fitur penting RandomForest:")
print(feat_importance_df.head(10))

# Visualisasi
plt.figure(figsize=(8,5))
plt.barh(feat_importance_df['feature'][:10], feat_importance_df['importance'][:10])
plt.gca().invert_yaxis()
plt.xlabel('Importances')
plt.title('Top 10 Feature Importances RandomForest')
plt.show()