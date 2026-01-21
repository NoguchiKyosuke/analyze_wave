import noisereduce as nr

nr.reduce_noise("F001_R1_E1_L1_BT.wav", "denoise.wav", freq_mask_smooth_hz=0, time_mask_smooth_ms=0)