import os
import numpy as np
import pandas as pd
import librosa
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- CONFIG ---
TARGET_SR = 16000
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048
MAX_FRAMES = 200

CLASS_LABELS = ['Bronchiectasis', 'Bronchiolitis', 'COPD', 'Healthy', 'Pneumonia', 'URTI']

DATA_DIR = "Respiratory_Sound_Database/Respiratory_Sound_Database"
CSV_PATH = os.path.join(DATA_DIR, "patient_diagnosis.csv")

EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 15
LR = 0.001

OUTPUT_DIR = "trained_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- DENOISING ---
def wavelet_denoise(y, wavelet='db8', level=5):
    coeffs = pywt.wavedec(y, wavelet, mode='per', level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs_thresh, wavelet, mode='per')


# --- FEATURE EXTRACTION ---
def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=-1)
    features = features.T
    D, T, C = features.shape
    if T < MAX_FRAMES:
        pad = MAX_FRAMES - T
        features = np.pad(features, ((0, 0), (0, pad), (0, 0)), mode='constant')
    elif T > MAX_FRAMES:
        features = features[:MAX_FRAMES, :, :]
    return features


def load_dataset(data_dir, csv_path, denoise=False):
    df = pd.read_csv(csv_path, header=None)
    id_to_label = dict(zip(df[0], df[1]))

    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_LABELS)

    X, y = [], []
    audio_dir = os.path.join(data_dir, "audio")
    if not os.path.exists(audio_dir):
        audio_dir = data_dir

    for fname in os.listdir(audio_dir):
        if not fname.endswith('.wav'):
            continue
        fpath = os.path.join(audio_dir, fname)
        patient_id = int(fname.split('_')[0])
        if patient_id not in id_to_label:
            continue

        wav, sr = librosa.load(fpath, sr=TARGET_SR)
        if denoise:
            wav = wavelet_denoise(wav)
        feats = extract_features(wav, sr)
        X.append(feats)
        y.append(patient_id)

    y = [id_to_label[pid] for pid in y]
    y = label_encoder.transform(y)

    return np.array(X), np.array(y), label_encoder


# --- MODELS ---
def cnn_bigru(input_shape=(MAX_FRAMES, N_MFCC, 3), num_classes=6):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='linear')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = layers.Reshape((50, 512))(x)
    x = layers.Bidirectional(layers.GRU(128))(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inp, out)


def cnn_lstm(input_shape=(MAX_FRAMES, N_MFCC, 3), num_classes=6):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='linear')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = layers.Conv2D(128, (3, 3), padding='same', activation='linear')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2), strides=2, padding='same')(x)

    x = layers.Reshape((50, 512))(x)
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(128)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inp, out)


# --- TRAINING ---
def train_model(model_fn, model_name, X_train, y_train, X_val, y_val):
    model = model_fn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(
            os.path.join(OUTPUT_DIR, f'{model_name}.keras'),
            monitor='val_accuracy', mode='max', save_best_only=True, verbose=1
        ),
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


# --- MAIN ---
if __name__ == '__main__':
    print("Loading original audio dataset...")
    X_orig, y_orig, le = load_dataset(DATA_DIR, CSV_PATH, denoise=False)
    print(f"Original: {X_orig.shape}, classes: {len(np.unique(y_orig))}")

    print("\nLoading denoised audio dataset...")
    X_den, y_den, _ = load_dataset(DATA_DIR, CSV_PATH, denoise=True)
    print(f"Denoised: {X_den.shape}, classes: {len(np.unique(y_den))}")

    datasets = [
        ('original', X_orig, y_orig),
        ('denoised', X_den, y_den),
    ]

    for label, X, y in datasets:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"\n{'='*60}")
        print(f"Training on {label} data")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")

        train_model(cnn_bigru, f'best_model_BiGRU_{label}',
                     X_train, y_train, X_val, y_val)
        train_model(cnn_lstm, f'best_model_CNN_LSTM_{label}',
                     X_train, y_train, X_val, y_val)

    print("\nAll training complete. Models saved to:", OUTPUT_DIR)
