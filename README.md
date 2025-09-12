# ML DDoS Detection

Kumpulan proyek machine learning untuk deteksi DDoS menggunakan network flow dataset. Repository ini berisi eksperimen dengan beberapa model machine learning untuk mengklasifikasikan trafik jaringan sebagai normal atau serangan.

> Machine Learning Project for Distributed Denial of Service (DDoS) attack detection on network traffic data. This repository contains experiments with different machine learning models to classify network flows as normal or attack.

---

## Dataset

- **Source**          : Network Traffic Dataset on Kaggle  
- **Size**            : 104,345 rows  
- **Features**        : dt, dur, dur_nsec, tot_dur, pktrate, Protocol, port_no, tx_kbps, rx_kbps, tot_kbps, label  
- **Target**          : 0 = Normal, 1 = DDoS  
- **Missing Values**  : 0.04%  

> Catatan: Missing values sangat sedikit dan sebagian algoritma (LightGBM, CatBoost, XGBoost) dapat menangani `NaN` secara native.

---

## Models Implemented

- **RandomForest Classifier**  
- **Multi-Layer Perceptron (MLP)**  
- **LightGBM (Gradient Boosting)**  
- **XGBoost**
- **CatBoost**

---

## Results

| Model                         | Accuracy |
|-------------------------------|--------- |
| RandomForest Classifier       | 99.6%    |
| Multi-Layer Perceptron (MLP)  | 98.8%    |
| LightGBM (Gradient Boosting)  | 99.9%    |
| XGBoost                       | 99.8%    |
| CatBoost                      | 99.8%    |

> Semua model telah dievaluasi menggunakan classification report (precision, recall, F1-score) dan confusion matrix. F1-score untuk kelas DDoS dan Normal sangat tinggi, menunjukkan model mampu generalisasi dengan baik.

---

## Feature Importance Insights

- **LightGBM** → fitur teratas: `dt` (duration)  
- **RandomForest** → fitur teratas: `pktrate`  
- **XGBoost** → fitur teratas: `protocol_ICMP`
- **CatBoost** → fitur teratas: `pktrate`

> Catatan: Fitur teratas berbeda-beda tergantung algoritma karena cara model menilai split/impurity berbeda. Semua top feature masih masuk akal secara domain: misalnya DDoS sering menunjukkan paket rate tinggi atau traffic ICMP abnormal.

---

## Notes

- LightGBM, XGBoost dan CatBoost secara native menangani missing values 0.04% tanpa imputasi tambahan.  
- XGBoost membutuhkan kolom kategori diubah menjadi numeric (OneHotEncoder).  
- CatBoost bisa langsung menangani kolom kategori tanpa OneHotEncoder dengan mendefinisikan kolom kategorialnya ke model dengan membuatkan variabel kategorial.  

---

