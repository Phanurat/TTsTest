from transformers import pipeline
import numpy as np
from scipy.io.wavfile import write

tts = pipeline(
    "text-to-speech",
    model="facebook/mms-tts-tha",
    device="cpu"
)

result = tts("ข้ามาที่นี่เพื่อเปลี่ยนประเทศนี้ให้กลายเป็นนรกที่มีชีวิต ข้าอยากจะทำให้ประเทศนี้เป็นประเทศที่สะท้อนเสียงร้อง คำสาป และเสียงกรีดร้องชั่วนิรันดร์")

audio = np.asarray(result["audio"], dtype=np.float32)
sr = int(result["sampling_rate"])

# 🔑 สำคัญที่สุด: flatten ให้เป็น mono 1D
audio = audio.squeeze()

# normalize
max_val = np.max(np.abs(audio))
if max_val > 0:
    audio = audio / max_val

audio = (audio * 32767).astype(np.int16)

write("thai_tts.wav", sr, audio)

print("Saved thai_tts.wav")
print("audio shape:", audio.shape)
