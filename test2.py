from pydub import AudioSegment

# Load the files
sound1 = AudioSegment.from_wav("combined.wav")
sound2 = AudioSegment.from_wav("F001_R1_E1_L1_BT.wav")

# Compare the files
if sound1 != sound2:
    print("Files do not match.")
else:
    print("Files match.")

diff = sound1.overlay(sound2.invert_phase())
diff.export("diff.wav", format="wav")