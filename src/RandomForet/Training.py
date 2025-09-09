Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol_TCP','Protocol_ICMP','Protocol_UDP',
    'port_no','tx_kbps','rx_kbps','tot_kbps'
]
df_clean = df[Features]
from sklearn.model_selection import train_test_split

X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size = 0.2, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)
model_rf.fit(X_train, y_train)
predict = model_rf.predict(X_test)
