from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

best_model = random_search.best_estimator_
# Prediksi di training set untuk mengetahui overfit sebuah model
y_train_pred = best_model.predict(X_train)

# Hitung akurasi
train_acc = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_acc)

# Evaluate model menggunakan datasheet test
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print('Akurasi :', acc)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X, y, cv=kf)
print(scores.mean())

# Korelasi fitur dengan label

corr = df[Features + ['label']].select_dtypes(include=['number']).corr()['label'].sort_values(ascending=False)
print(corr)

# Grafik evaluasi confusion matrix

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

preprocessor  = best_model.named_steps['preprocess']
model = best_model.named_steps['classifier']

# Nama kolom dari masing-masing transformer
categorical_features = ['Protocol']
numeric_features = ['dt','dur','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']

# Pastikan preprocessor sudah fit (otomatis sudah karena RandomSearchCV fit)
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
num_features = numeric_features

feature_names = list(cat_features) + list(num_features)

# Top Fitur XGBoost
importances = model.feature_importances_

feat_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

print(feat_importance_df.head(15))