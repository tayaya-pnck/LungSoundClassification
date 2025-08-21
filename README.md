# 🫁 Lung Sound Classification using CNN-LSTM & CNN-BiGRU
This project focuses on lung sound classification to assist in the early diagnosis of respiratory diseases using deep learning. It utilizes ICBHI 2017 Respiratory Sound Database and implements hybrid deep learning architectures combining Convolutional Neural Networks (CNN) with LSTM and Bidirectional GRU (BiGRU) for temporal feature learning.
# 📌 Overview
Respiratory diseases such as COPD, Bronchiectasis, Bronchiolitis, URTI, and Pneumonia require accurate and timely diagnosis. This project leverages audio processing techniques and deep learning models to classify lung sounds into multiple categories.

We compare two models on two different datasets:

Original Lung Sound Data

Denoised Lung Sound Data (processed using adaptive filtering and wavelet denoising)

The models evaluated:

✅ CNN-LSTM (Convolutional + Long Short-Term Memory)

✅ CNN-BiGRU (Convolutional + Bidirectional Gated Recurrent Unit)
# ⚙️ Features
Audio Preprocessing:

MFCC, Delta, and Delta-Delta feature extraction

Noise reduction using Wavelet Denoising & EMD-based adaptive filtering

Model Architectures:

CNN-LSTM for spatiotemporal feature extraction

CNN-BiGRU for improved sequential modeling

Comparison between Original vs Denoised Data

Metrics: Accuracy, Precision, Recall, F1-Score

Implemented in TensorFlow/Keras

# 📊 Results
| Model     | Dataset  | Accuracy   | Precision  | Recall     | F1-Score   |
| --------- | -------- | ---------- | ---------- | ---------- | ---------- |
| CNN-LSTM  | Original | 92.03%     | 0.7593     | 0.9846     | 0.8312     |
| CNN-LSTM  | Denoised | 72.46%     | 0.3830     | 0.7038     | 0.4461     |
| CNN-BiGRU | Original | 95.65%     | 0.8444     | 0.9916     | 0.9022     |
| CNN-BiGRU | Denoised | **97.10%** | **0.8750** | **0.9944** | **0.9249** |

✅ Best Performance: CNN-BiGRU with Denoised Data (97.10% accuracy)

# 🏗️ Model Architecture
CNN Layers: Extract spatial features from spectrogram input

Pooling & Batch Normalization: Stabilize and reduce dimensions

Reshape Layer: Convert feature maps for sequential layers

Recurrent Layers:

LSTM for temporal dependencies

BiGRU for bidirectional context learning

Dense Layers: Fully connected layers for classification

# 📦 Tech Stack
Python 3.x

TensorFlow / Keras

NumPy, Pandas

Librosa (for audio processing)

Matplotlib, Seaborn (for visualization)

# 🖊️ Author

Athaya Abdan Hanif
🔥 If you like this project, give it a ⭐ on GitHub and share it!
