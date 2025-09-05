# ML DDoS Detection
Kumpulan proyek machine learning deteksi DDoS dengan berbagai model yang digunakan dan datasheet yang sama bahkan berbeda. 
> Machine Learning Project for Distributed Denial of Service (DDoS) attack detection on network traffic data. This repository contains experiments with different machine learning models to classify network flows as normal or attack

## Dataset
Source    : Network Traffic Dataset on Kaggle
Size      : 104345 rows
Features  : dt, dur, dur_nsec, tot_dur, pktrate, Protocol, port_no, tx_kbps, rx_kbps, tot_kbps, label
Target    : 0 = Normal, and 1 = DDoS

## Models Implemented
RandomForest Classifier

## Results
RandomForest Classifier accuracy : 99.6%
