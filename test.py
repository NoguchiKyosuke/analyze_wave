from pydub import AudioSegment

# Load the files
sound1 = AudioSegment.from_wav("F001_R1_E1_L1_BT_clean.wav")
sound2 = AudioSegment.from_wav("F001_R1_E1_L1_BT_noise.wav")

# Overlay (mix) them
combined = sound1.overlay(sound2)

# Export the result
combined.export("combined.wav", format="wav")