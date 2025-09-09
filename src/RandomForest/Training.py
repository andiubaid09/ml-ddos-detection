from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced' #Otomatis beri bobot pada kelas minoritas
)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
