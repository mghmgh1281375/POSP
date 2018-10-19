"""
# Mohammad Ghorbani -- 94470823
---
 1. Record sound array.
 2. Save sound array.
 3. Read sound array.
 4. Plot spectrogram figure.
"""

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# frame per second in most cases will be 44100 or 48000
fs = 44100

# Setting sample rate (frame per second)
sd.default.samplerate = fs

# Setting number of channels
sd.default.channels = 2

# Recording duration
duration = 1  # seconds

myrecording = sd.rec(int(duration * fs), dtype='float64')

# This command waits untill recording ends.
sd.wait()

# Saving sound array
np.save('./three.wav', myrecording)

# Loading sound array
myrecording = np.load('./three.wav.npy')

# Playing sound array
sd.play(myrecording)

# Ploting spectrogram of sound array
fig = plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.title('First Channel')
plt.specgram(myrecording[:, 0], Fs=fs)
plt.subplot(1, 2, 2)
plt.title('Second Channel')
plt.specgram(myrecording[:, 1], Fs=fs)
plt.show()
print(myrecording.shape)
print(myrecording)