from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

best_modelCAT = random_search.best_estimator_
print(best_modelCAT)

# Pengujian menggunakan dataset training

y_pred_train = best_modelCAT.predict(X_train)
acc = accuracy_score(y_train, y_pred_train)
print(f"Akurasi model dengan data training : {acc:.4f}")

# Pengujian menggunakan dataset uji

y_predi_test = best_modelCAT.predict(X_test)
acc_test = accuracy_score(y_test, y_predi_test)
print(f"Akurasi model dengan data pengujian : {acc_test:.4f}")
print(classification_report(y_test,y_predi_test))

# Validasi akurasi dengan cross validation

kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = best_modelCAT
    model.fit(X_train, y_train, classifier__cat_features=[len(Numeric_features)])
    scores.append(model.score(X_test, y_test))

print("Mean CV Accuracy:", np.mean(scores))
cm = confusion_matrix(y_test, y_predi_test)
print(cm)

# Visualisasi 

label = ['Normal', 'DDoS']
cm = confusion_matrix(y_test, y_predi_test)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap = 'Blues',
            xticklabels= label,
            yticklabels= label)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Top Fitur
cat_model = best_modelCAT.named_steps['classifier'] # Ambil model CATBOOST dari Pipeline
preprocessor_fit = best_modelCAT.named_steps['preprocess']
feature_importances = cat_model.get_feature_importance()
all_features = preprocessor_fit.get_feature_names_out()

importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10))
top_features = importance_df.head(10)

plt.figure(figsize=(10,6))
plt.barh(top_features['Feature'],top_features['Importance'])
plt.gca().invert_yaxis() # Rangking tertinggi di atas
plt.title('Top 10 Features Importance CatBoost')
plt.xlabel('Importances')
plt.ylabel('Feature')
plt.grid(axis= 'x', linestyle='--', alpha=0.7)
plt.show()

# Korelasi Fitur
corr = df[Features + ['label']].select_dtypes(include=['number']).corr()['label'].sort_values(ascending=False)
print(corr)
