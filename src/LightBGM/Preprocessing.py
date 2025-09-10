import pandas as pd
import numpy as np
from google.colab import drive
drive.mount ('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/dataset_sdn.csv'
df = pd.read_csv(datasheet)
# Cek Missing Values Kolom
print("Jumlah Missing Values :")
print(df.isnull().sum())
# Drop Missing Values
df = df.dropna(subset=['rx_kbps','tot_kbps'])
print(df.isnull().sum())
# Melakukan penanganan fitur kategorikal dengan mengubah menjadi binary menggunakan OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

df['Protocol'] = df ['Protocol'].astype(str)
encoder = OneHotEncoder(sparse_output=False)
encoder_clean = encoder.fit_transform(df[['Protocol']]).astype(int)
encoder_clean = pd.DataFrame(
    encoder_clean,
    columns= encoder.get_feature_names_out(['Protocol']),
    index= df.index
)
df = pd.concat([df.drop(columns=['Protocol']), encoder_clean], axis=1)
#Melihat kolom yang telah di OneHotEncoder
df.columns
# Memilih fitur yang relevan dalam mendeteksi DDoS
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol_ICMP','Protocol_TCP','Protocol_UDP','port_no',
    'tx_kbps','rx_kbps','tot_kbps'
]
df_clean = df[Features]
# Membagi data dengan komposisi X = 80% dan y = 20% untuk model
from sklearn.model_selection import train_test_split

X = df_clean
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42, stratify=y
)
# Menghitung rasio untuk imbalance
ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
print("Ratio class imbalance (0/1)",ratio)

#Hasilnya akan menunjukkan Kelas 0(Normal) ada 1,56x lebih banyak daripada kelas 1(DDoS) di training set
