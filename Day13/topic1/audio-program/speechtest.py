#%%
import librosa
import audioread
import numpy as np 
import matplotlib.pyplot as plt
import librosa.display
from scipy.io import wavfile

#%%
inputfile = "test1.wav"
outputfile = "testout1.wav"
outputfile2 = "testout2.wav"

#%%
# Read in file with audioread
with audioread.audio_open(inputfile) as f:
    print(f.channels, f.samplerate, f.duration)
#%%
# Read in file with librosa
wav, sr = librosa.core.load(inputfile, sr=16000)

#%%
# Normalization
wav *= 1 / max(0.01, np.max(np.abs(wav))) 

# Change sampling rate
wav_8k = librosa.resample(wav, sr, 8000)

#%%
# Save wave file with scipy
wavfile.write(outputfile, sr, wav.astype(np.int16))
wavfile.write(outputfile2, sr, wav_8k.astype(np.int16))

#%%
# Read in a sample file
y, sr = librosa.load(librosa.util.example_audio_file())

#%%
# Display waveform
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Monophonic')

#%% Calculate stft feature
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

#%%
# Display spectrogram in linear frequency scale
plt.figure()
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

#%%
# Display spectrogram in log frequency scale
plt.figure()
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')
