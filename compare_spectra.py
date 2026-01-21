import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, spectrogram
import os

def compare_spectra(file_list, output_filename="comparison_spectra.png"):
    """
    Compares the Power Spectral Density (PSD) of multiple audio files.
    """
    plt.figure(figsize=(12, 8))
    
    # Setup the plot
    ax = plt.subplot(1, 1, 1)
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    for i, file_path in enumerate(file_list):
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
            
        try:
            rate, data = wavfile.read(file_path)
            
            # Handle stereo by taking the first channel
            if len(data.shape) > 1:
                data = data[:, 0]
                
            # Compute PSD using Welch's method
            # nperseg determines the frequency resolution
            f, Pxx = welch(data, rate, nperseg=4096)
            
            # Label from filename
            label = os.path.basename(file_path)
            
            # Plot
            # Using semi-log Y scale is standard for PSD
            ax.semilogy(f, Pxx, label=label, alpha=0.7, color=colors[i % len(colors)])
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    ax.set_title('Power Spectral Density Comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (V**2/Hz)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(0, 8500) # Typical audio range
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Comparison plot saved to {output_filename}")

def compare_spectrograms(file_list, output_filename="comparison_spectrograms.png"):
    """
    Generates and compares spectrograms (Time vs Frequency) for multiple audio files.
    """
    num_files = len(file_list)
    plt.figure(figsize=(15, 5 * num_files))
    
    for i, file_path in enumerate(file_list):
        if not os.path.exists(file_path):
            continue
            
        try:
            rate, data = wavfile.read(file_path)
            
            # Handle stereo
            if len(data.shape) > 1:
                data = data[:, 0]
            
            # Subplot layout: num_files rows, 1 column
            ax = plt.subplot(num_files, 1, i + 1)
            
            # Compute spectrogram
            # nperseg=1024 for a good balance of time/freq resolution for speech
            f, t, Sxx = spectrogram(data, rate, nperseg=1024)
            
            # Plot using pcolormesh and log scale for intensity (dB)
            # Add a small epsilon to avoid log(0)
            img = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
            
            label = os.path.basename(file_path)
            ax.set_title(f'Spectrogram: {label}')
            ax.set_ylabel('Frequency (Hz)')
            
            if i == num_files - 1:
                ax.set_xlabel('Time (s)')
            
            # Add colorbar
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Spectrogram comparison saved to {output_filename}")

if __name__ == "__main__":
    # The specific files requested by the user
    files_to_compare = [
        "/home/nk21137/OneDrive/5years/graduation_research/analyze_wave/M1/M001_R2_E4_L1_BU.wav",
        "/home/nk21137/OneDrive/5years/graduation_research/analyze_wave/M1/M001_R2_E4_L2_BU.wav",
        "/home/nk21137/OneDrive/5years/graduation_research/analyze_wave/M1/M001_R2_E4_M1_BU.wav"
    ]
    
    # Check if files exist, if not try relative paths
    base_dir = "/home/nk21137/OneDrive/5years/graduation_research/analyze_wave/"
    resolved_files = []
    for f in files_to_compare:
        if os.path.exists(f):
            resolved_files.append(f)
        elif os.path.exists(os.path.join(base_dir, os.path.basename(f))):
             resolved_files.append(os.path.join(base_dir, os.path.basename(f)))
        else:
             print(f"Could not find file: {f}")

    if resolved_files:
        compare_spectra(resolved_files, "F001_spectra_comparison.png")
        compare_spectrograms(resolved_files, "F001_spectrogram_comparison.png")
