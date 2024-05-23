import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def split_and_save_audio(file_path, output_dir, segment_length=5):
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    segment_samples = int(segment_length * sr)
    file_name = os.path.basename(file_path).split('.')[0]
    segments = []
    num_segments = len(audio) // segment_samples
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]
        stereo_segment = np.vstack((segment, segment))  # Mono segmenti stereo segmente çevir
        segment_file_name = f"{file_name}_segment_{i}.wav"
        segment_file_path = os.path.join(output_dir, segment_file_name)
        sf.write(segment_file_path, stereo_segment.T, sr)  # Transpose yaparak stereo olarak kaydet
        segments.append(segment_file_path)
    return segments

def features_extractor(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=True)  # Mono olarak yükleyin
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

def klasorleri_gez_ve_sesleri_bol_ve_ozellikleri_cikar(klasorler):
    ozellikler = []
    etiketler = []
    for klasor in klasorler:
        dosya_yolu = os.path.join(klasor, klasor + ".wav")
        output_dir = os.path.join(klasor, "segments")
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isfile(dosya_yolu):
            segments = split_and_save_audio(dosya_yolu, output_dir)
            for segment_file_path in segments:
                ozellik = features_extractor(segment_file_path)
                ozellikler.append(ozellik)
                etiketler.append(klasor)
    return np.array(ozellikler), np.array(etiketler)

klasorler = ["bora","eda","gul","mithat"]

X, y = klasorleri_gez_ve_sesleri_bol_ve_ozellikleri_cikar(klasorler)

print("Elde edilen özelliklerin şekli ve boyutu:", X.shape)
print("Elde edilen etiketlerin boyutu:", y.shape)

labelencoder = LabelEncoder()
y_encoded = labelencoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, alpha=0.001, 
                    solver='adam', verbose=True, random_state=21, tol=1e-9)

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=labelencoder.classes_))
print(f"Test Accuracy: {accuracy}")

joblib.dump(mlp, 'mlp_ses_modeli.pkl')
joblib.dump(labelencoder, 'label_encoder.pkl')

def predict_audio_class(file_path, model, labelencoder, segment_length=5):
    temp_dir = "temp_segments"
    os.makedirs(temp_dir, exist_ok=True)
    segments = split_and_save_audio(file_path, temp_dir, segment_length)
    predictions = []
    for segment_file_path in segments:
        mfccs_scaled_features = features_extractor(segment_file_path)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        predicted_label = model.predict(mfccs_scaled_features)
        predictions.append(predicted_label[0])
    prediction_class = labelencoder.inverse_transform(predictions)
    return prediction_class

prediction = predict_audio_class("gul/segments/gul_segment_4.wav", mlp, labelencoder)
print("Tahmin edilen sınıf:", prediction)