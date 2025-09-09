import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

datasheet = '/content/drive/My Drive/Datasheet/dataset_sdn.csv'

df = pd.read_csv(datasheet)
df.info()
print("Jumlah Missing Values")
print(df.isnull().sum())
df = df.dropna(subset=['tx_kbps','tot_kbps'])
df['label'] = df['label'].astype(int)
df.info()

from sklearn.preprocessing import OneHotEncoder

df['Protocol'] = df['Protocol'].astype(str)

encoder = OneHotEncoder(sparse_output=False)
encoded_clean = encoder.fit_transform(df[['Protocol']]).astype(int)
encoded_clean = pd.DataFrame(
    encoded_clean,
    columns=encoder.get_feature_names_out(['Protocol']),
    index=df.index #Baris tetap sinkron
)
df = pd.concat([df.drop(columns=['Protocol']), encoded_clean], axis=1)
df.info()
Features = [
    'dt','dur','dur_nsec','tot_dur','pktrate','Protocol_ICMP','Protocol_TCP','Protocol_UDP',
    'port_no','tx_kbps','rx_kbps','tot_kbps'
]

from sklearn.model_selection import train_test_split

x = df_clean[Features]
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
