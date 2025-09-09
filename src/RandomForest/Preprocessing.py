import pandas as pd
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

file_path = '/content/drive/My Drive/Skripsi/Datasheet/DH_label_only.csv'

df_label = pd.read_csv(file_path)
df_label.info()
df_label['label'].value_counts()
print("Jumlah Missing Values:")
print(df_label.isnull().sum())
df_label = df_label.dropna(subset=['tx_kbps', 'tot_kbps'])
df_label['label'] = df_label['label'].astype(int)
df_label.info()

from sklearn.preprocessing import OneHotEncoder

df_label['Protocol'] = df_label['Protocol'].astype(str)

encoder = OneHotEncoder(sparse_output=False)
encoded_clean = encoder.fit_transform(df_label[['Protocol']]).astype(int)
encoded_clean = pd.DataFrame(
    encoded_clean,
    columns=encoder.get_feature_names_out(['Protocol']),
    index=df_label.index #Baris tetap sinkron
)
df_label = pd.concat([df_label.drop(columns=['Protocol']), encoded_clean], axis=1)
df_label.info()

Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol_ICMP','Protocol_TCP','Protocol_UDP',
    'port_no','tx_kbps','rx_kbps','tot_kbps','label'
]
df_clean = df_label[Features]
df_clean.info()

from sklearn.model_selection import train_test_split

x = df_clean.drop('label', axis=1)
y = df_clean['label']
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)
