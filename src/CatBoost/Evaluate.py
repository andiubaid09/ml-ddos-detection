from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

y_pred = modelCat.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(f'Akurasi : {acc:.4f}')

import matplotlib.pyplot as plt
import seaborn as sns

label = ['Normal','DDoS']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels = label,
            yticklabels = label)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

#Top Fitur
test_pool = Pool(X_test, y_test, cat_features=categorical_cols)
feature_importances = modelCat.get_feature_importance(test_pool)

feat_imp =pd.DataFrame({
    "Feature" : X_test.columns,
    "Importance": feature_importances
}).sort_values(by="Importance",ascending=False)
print(feat_imp)
plt.figure(figsize=(10,8))
sns.barplot(data=feat_imp, x="Importance", y='Feature', palette='viridis')
plt.title('Fitur Penting (CatBoost)')
plt.show()

from google.colab import files
modelCat.save_model('catboost_detection_DDoS.cbm')
files.download('catboost_detection_DDoS.cbm')
