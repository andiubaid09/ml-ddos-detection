from google.colab import drive
import pandas as pd

drive.mount('/content/drive')
datasheet = '/content/drive/My Drive/Datasheet/DDoS/dataset_sdn.csv'
df = pd.read_csv(datasheet)
print('Jumlah Missing Values')
print(df.isnull().sum())

# Menghapus kolom bernilai null
df = df.dropna(subset=['rx_kbps','tot_kbps'])
df.info()
Features = ['dt','dur_nsec','tot_dur','pktrate','Protocol','port_no','tx_kbps','rx_kbps','tot_kbps']
df_feat = df[Features]

from sklearn.model_selection import train_test_split
X = df_feat
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ['dt','dur_nsec','tot_dur','pktrate','port_no','tx_kbps','rx_kbps','tot_kbps']
kateogrial_features = ['Protocol']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_cat = kateogrial_features
encoder_transform = encoder
scaler = StandardScaler()
numeric_feat = numeric_features
numeric_transform = scaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat',encoder_transform, encoder_cat),
        ('num',numeric_transform, numeric_feat)
    ],
    remainder='drop'
)

# Definisi base model
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced',
    n_jobs= -1
)
rf_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', model_rf)
])