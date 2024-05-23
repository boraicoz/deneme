import os
import tkinter as tk
from tkinter import messagebox, Label, Button, Tk
import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import joblib
import speech_recognition as sr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

mlp = joblib.load('C:\\Users\\borai\\OneDrive\\Belgeler\\Proje\\mlp_ses_modeli.pkl')
labelencoder = joblib.load('C:\\Users\\borai\\OneDrive\\Belgeler\\Proje\\label_encoder.pkl')

sample_rate = 44100
channels = 1
segment_duration = 5
save_dir = "C:\\Users\\borai\\OneDrive\\Belgeler\\Proje\\Kayıtlarımız"
current_recording_index = 1

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def transcribe_audio(audio, sample_rate):
    recognizer = sr.Recognizer()
    audio = np.squeeze(audio)

    audio = (audio * 32767).astype(np.int16)
    audio_bytes = audio.tobytes()

    audio_data = sr.AudioData(audio_bytes, sample_rate, 2)

    try:
        transcript = recognizer.recognize_google(audio_data, language="tr-TR")
        return transcript
    except sr.UnknownValueError:
        return "Google Web Speech API sesi anlayamadı."
    except sr.RequestError as e:
        return f"Google Web Speech API'den sonuç alınamadı; {e}"

def predict_from_mic(audio, result_window):
    global current_recording_index
    
    mfccs_features = librosa.feature.mfcc(y=np.squeeze(audio), sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)

    predicted_label = mlp.predict(mfccs_scaled_features)
    predicted_class = labelencoder.inverse_transform([predicted_label])[0]

    transcript = transcribe_audio(audio, sample_rate)

    if transcript not in ["Google Web Speech API sesi anlayamadı.", "Google Web Speech API'den sonuç alınamadı;"]:
        result_window.speaker_value.config(text=predicted_class)
        result_window.words_value.config(text=transcript)
        result_window.total_words_value.config(text=len(transcript.split()))

        audio = np.squeeze(audio)
        fig, ax = plt.subplots()
        ax.hist(audio, bins=50, color='blue', alpha=0.7)
        ax.set_title('Ses Histogramı')
        ax.set_xlabel('Genlik')
        ax.set_ylabel('Frekans')

        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().grid(row=3, column=0, columnspan=2, pady=10)

        # Ses kaydını kaydet
        save_path = os.path.join(save_dir, f"Kayıt{current_recording_index}.wav")
        sf.write(save_path, audio, sample_rate)
        current_recording_index += 1

        messagebox.showinfo("Bilgi", "Sesiniz Kayıtlarımız Kısmına Kaydedildi.")

def exit_application():
    root.destroy()

def show_recordings():
    files = os.listdir(save_dir)
    if files:
        recordings_window = tk.Toplevel(root)
        recordings_window.title("Kayıtlar")
        
        tk.Label(recordings_window, text="Kayıtlı Ses Dosyaları:", font=("Arial", 12, "bold")).pack(pady=10)

        for file in files:
            file_path = os.path.join(save_dir, file)
            tk.Label(recordings_window, text=file, font=("Arial", 12)).pack()
    else:
        messagebox.showinfo("Bilgi", "Hiç Kayıtlı Ses Dosyası Yok.")

class SoundRecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sound Recorder")
        self.root.geometry("800x600")
        self.root.minsize(300, 300)
        self.root.configure(background="green")

        self.welcome_label = tk.Label(root, text="SES ALGILAYICI", font=("Georgia", 36, "bold"), pady=10, background="green", fg="aqua")
        self.welcome_label.pack()

        self.record_button = tk.Button(root, text="Ses Kaydet", command=self.record_sound, font=("Arial", 15), bg="orange", fg="black", width=15)
        self.record_button.pack(ipady=10, ipadx=10, padx=30, pady=20)

        self.predict_button = tk.Button(root, text="Tahmin Et", command=self.predict_sound, font=("Arial", 15), bg="orange", fg="black", width=15, state=tk.DISABLED)
        self.predict_button.pack(ipady=10, ipadx=10, padx=30, pady=20)

        self.recordings_button = tk.Button(root, text="Kayıtlar", command=show_recordings, font=("Arial", 15), bg="orange", fg="black", width=15)
        self.recordings_button.pack(ipady=10, ipadx=10, padx=30, pady=20)
        
        self.exit_button = tk.Button(root, text="Çıkış", command=exit_application, font=("Arial", 15), bg="orange", fg="black", width=15)
        self.exit_button.pack(ipady=10, ipadx=10, padx=30, pady=20)

        self.result_window = None
        self.recording = None

    def record_sound(self):
        fs = 44100
        duration = 5
        self.recording = sd.rec(int(fs * duration), samplerate=fs, channels=1)
        sd.wait()
        self.predict_button.config(state=tk.NORMAL)

    def predict_sound(self):
        if self.recording is not None:
            if self.result_window:
                self.result_window.destroy()
            self.result_window = tk.Toplevel(self.root)
            self.result_window.title("Tahmin Sonuçları")

            speaker_label = Label(self.result_window, text="Konuşan Kişi:", font=("Arial", 12))
            speaker_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")

            self.result_window.speaker_value = Label(self.result_window, text="", font=("Arial", 12, "bold"))
            self.result_window.speaker_value.grid(row=0, column=1, padx=10, pady=5, sticky="w")

            words_label = Label(self.result_window, text="Söylenen Kelimeler:", font=("Arial", 12))
            words_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")

            self.result_window.words_value = Label(self.result_window, text="", font=("Arial", 12, "bold"))
            self.result_window.words_value.grid(row=1, column=1, padx=10, pady=5, sticky="w")
            total_words_label = Label(self.result_window, text="Kullanılan Toplam Kelime Sayısı:", font=("Arial", 12))
            total_words_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")

            self.result_window.total_words_value = Label(self.result_window, text="", font=("Arial", 12, "bold"))
            self.result_window.total_words_value.grid(row=2, column=1, padx=10, pady=5, sticky="w")

            predict_from_mic(self.recording, self.result_window)
            self.recording = None
            self.predict_button.config(state=tk.DISABLED)
            self.record_button.config(state=tk.NORMAL)  # This line enables the Record button
        else:
            messagebox.showwarning("Uyarı", "Lütfen Önce Bir Ses Kaydedin.")

if __name__ == "__main__":
    root = tk.Tk()
    app = SoundRecorderGUI(root)
    root.mainloop()
