from gtts import gTTS
import os

# Text to be spoken
text = "Hello, could you please tell me a short story about a magical forest?"

# Create gTTS object
tts = gTTS(text=text, lang='en', slow=False)

# Save the audio file
tts.save('test_audio.wav')

print("Test audio file generated: test_audio.wav")
print(f"Text content: '{text}'") 