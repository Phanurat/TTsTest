from transformers import pipeline
import numpy as np
from scipy.io.wavfile import write
import librosa
from scipy.signal import butter, lfilter

tts = pipeline("text-to-speech", model="facebook/mms-tts-tha", device="cpu")

result = tts("สวัสดีค่ะ ระบบนี้กำลังทดสอบการปรับโทนเสียง")

audio = np.asarray(result["audio"], dtype=np.float32).squeeze()
sr = int(result["sampling_rate"])

# pitch ↑
audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)

# speed ↑
audio = librosa.effects.time_stretch(audio, rate=1.05)

# soft voice
def lowpass(data, cutoff=3200, sr=16000):
    from scipy.signal import butter, lfilter
    b, a = butter(6, cutoff / (0.5 * sr), btype='low')
    return lfilter(b, a, data)

audio = lowpass(audio, cutoff=3200, sr=sr)

# normalize
audio /= np.max(np.abs(audio))
audio = (audio * 32767).astype(np.int16)

write("thai_tts_female_soft.wav", sr, audio)
