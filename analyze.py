
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, welch
import os

def analyze_audio(file_info):
    # Setup for main analysis
    plt.figure(figsize=(15, 12))
    num_files = len(file_info)
    
    # Store data for spectral slice analysis
    audio_data_list = []
    
    # 1. Main Spectrogram and Waveform Plots
    for i, (filename, label) in enumerate(file_info):
        if not os.path.exists(filename):
            print(f"Warning: File {filename} not found.")
            audio_data_list.append(None)
            continue
            
        rate, data = wavfile.read(filename)
        
        # If stereo, take just one channel for simplicity
        if len(data.shape) > 1:
            data = data[:, 0]
            
        time = np.arange(len(data)) / rate
        
        # Store for later
        audio_data_list.append({'rate': rate, 'data': data, 'time': time, 'label': label, 'filename': filename})
        
        # Waveform
        ax_wave = plt.subplot(num_files, 2, 2*i + 1)
        ax_wave.plot(time, data, alpha=0.7)
        ax_wave.set_title(f'{label} - Waveform')
        ax_wave.set_ylabel('Amplitude')
        ax_wave.set_xlabel('Time (s)')
        ax_wave.grid(True, alpha=0.3)
        
        # Spectrogram
        ax_spec = plt.subplot(num_files, 2, 2*i + 2)
        f, t, Sxx = spectrogram(data, rate)
        # Use log scale for better visibility
        img = ax_spec.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        ax_spec.set_title(f'{label} - Spectrogram')
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_xlabel('Time (s)')
        plt.colorbar(img, ax=ax_spec, format='%+2.0f dB')

    plt.tight_layout()
    output_filename = 'audio_analysis_comparison.png'
    plt.savefig(output_filename)
    print(f"Main analysis complete. Saved to {output_filename}")

    # 2. Spectral Analysis at Specific Moments
    run_spectral_moments_analysis(audio_data_list)
    
    # 3. Noise Reduction Visualization (New)
    visualize_denoising_process(audio_data_list)

def run_spectral_moments_analysis(audio_data_list):
    if not audio_data_list or audio_data_list[0] is None:
        return

    # Determine moments to analyze based on the first file (Original)
    ref_data = audio_data_list[0]
    duration = ref_data['time'][-1]
    
    # Pick moments: Peak amplitude, 1/4 duration, 3/4 duration
    peak_idx = np.argmax(np.abs(ref_data['data']))
    peak_time = ref_data['time'][peak_idx]
    
    moments = [
        duration * 0.25,
        peak_time,
        duration * 0.75
    ]
    moment_labels = ["25% Duration", "Peak Amplitude", "75% Duration"]
    
    fft_window_size = 2048 # Samples for FFT window
    
    plt.figure(figsize=(18, 6))
    
    for m_idx, (moment_time, m_label) in enumerate(zip(moments, moment_labels)):
        ax = plt.subplot(1, 3, m_idx + 1)
        
        for item in audio_data_list:
            if item is None: continue
            
            rate = item['rate']
            data = item['data']
            
            # Find index corresponding to moment
            center_idx = int(moment_time * rate)
            start_idx = max(0, center_idx - fft_window_size // 2)
            end_idx = min(len(data), center_idx + fft_window_size // 2)
            
            if end_idx - start_idx < fft_window_size:
                continue # Skip if near edges and not enough data
                
            segment = data[start_idx:end_idx]
            
            # Apply window function
            window = np.hamming(len(segment))
            spectrum = np.fft.rfft(segment * window)
            freqs = np.fft.rfftfreq(len(segment), 1/rate)
            
            magnitude = 20 * np.log10(np.abs(spectrum) + 1e-10)
            
            # Plot
            ax.plot(freqs, magnitude, label=item['label'], alpha=0.8)
            
        ax.set_title(f'Spectrum at {m_label} ({moment_time:.2f}s)')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude (dB)')
        # Limit to relevant audio frequencies (e.g., up to 8kHz or Nyquist)
        ax.set_xlim(0, 8000) 
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    bar_output_filename = 'spectral_comparison_moments.png'
    plt.savefig(bar_output_filename)
    print(f"Spectral comparison complete. Saved to {bar_output_filename}")

def visualize_denoising_process(audio_data_list):
    """
    Visualizes the difference between Original and Clean signals to show what was removed.
    """
    if len(audio_data_list) < 2:
        return
        
    original = audio_data_list[0]
    clean = audio_data_list[1]
    
    if original is None or clean is None:
        return

    # Ensure lengths match for subtraction
    min_len = min(len(original['data']), len(clean['data']))
    orig_data = original['data'][:min_len].astype(np.float64)
    clean_data = clean['data'][:min_len].astype(np.float64)
    rate = original['rate']
    
    # Calculate Residual (Noise Removed)
    residual_data = orig_data - clean_data
    
    plt.figure(figsize=(15, 10))
    
    # 1. Spectrogram: Original
    ax1 = plt.subplot(3, 2, 1)
    f, t, Sxx = spectrogram(orig_data, rate)
    ax1.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax1.set_title('1. Original Signal Spectrogram')
    ax1.set_ylabel('Frequency (Hz)')
    
    # 2. Spectrogram: Clean
    ax2 = plt.subplot(3, 2, 3)
    f, t, Sxx = spectrogram(clean_data, rate)
    ax2.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax2.set_title('2. Cleaned Signal Spectrogram')
    ax2.set_ylabel('Frequency (Hz)')
    
    # 3. Spectrogram: Residual (What was removed)
    ax3 = plt.subplot(3, 2, 5)
    f, t, Sxx = spectrogram(residual_data, rate)
    ax3.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
    ax3.set_title('3. Residual (Removed Noise) Spectrogram')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.set_xlabel('Time (s)')
    
    # 4. PSD Comparison (Original vs Clean vs Residual)
    ax4 = plt.subplot(1, 2, 2)
    
    f_orig, Pxx_orig = welch(orig_data, rate, nperseg=2048)
    f_clean, Pxx_clean = welch(clean_data, rate, nperseg=2048)
    f_res, Pxx_res = welch(residual_data, rate, nperseg=2048)
    
    ax4.semilogy(f_orig, Pxx_orig, label='Original', alpha=0.7)
    ax4.semilogy(f_clean, Pxx_clean, label='Clean', alpha=0.7)
    ax4.semilogy(f_res, Pxx_res, label='Residual (Diff)', alpha=0.7, linestyle='--')
    
    ax4.set_title('Power Spectral Density Comparison')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('PSD (V**2/Hz)')
    ax4.legend()
    ax4.grid(True, which='both', alpha=0.3)
    ax4.set_xlim(0, 10000) # Focus on relevant range
    
    plt.tight_layout()
    output_filename = 'denoising_process.png'
    plt.savefig(output_filename)
    print(f"Denoising process visualization saved to {output_filename}")


if __name__ == "__main__":
    # List of tuples: (filename, label)
    files_to_analyze = [
        ("F001_R1_E1_M1_BT.wav", "Original Audio"),
        ("F001_R1_E1_M1_BT_clean.wav", "Cleaned (Denoised)"),
        ("F001_R1_E1_M1_BT_noise.wav", "Extracted Noise")
    ]
    analyze_audio(files_to_analyze)
