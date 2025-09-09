Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol_ICMP','Protocol_TCP','Protocol_UDP',
    'port_no','tx_kbps','rx_kbps','tot_kbps','label'
]
df_clean = df_label[Features]
df_clean.info()
df_clean['label'].value_counts()

from sklearn.model_selection import train_test_split

x = df_clean.drop('label', axis=1)
y = df_clean['label']
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced' #Otomatis beri bobot pada kelas minoritas
)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
