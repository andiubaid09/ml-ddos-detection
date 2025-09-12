import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/dataset_sdn.csv'
df = pd.read_csv(datasheet)
df.info()

#Mengubah kolom kategorial Protokol menjadi string atau object
categorical_cols = ['Protocol']
for col in categorical_cols:
  df[col] = df[col].astype(str)

Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol','port_no','tx_kbps',
    'rx_kbps','tot_kbps'
]
df_clean = df[Features]

from sklearn.model_selection import train_test_split

X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)
