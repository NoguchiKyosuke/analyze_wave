import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import welch, spectrogram, stft
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

def compare_phase_spectrograms(file_list, output_filename="comparison_phase_spectrograms.png"):
    """
    Generates and compares phase spectrograms for multiple audio files.
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
            
            # Compute STFT to get complex values
            f, t, Zxx = stft(data, rate, nperseg=1024)
            
            # Extract angle (phase)
            phase = np.angle(Zxx)
            
            # Plot using pcolormesh
            # Phase is between -pi and pi
            img = ax.pcolormesh(t, f, phase, shading='gouraud', cmap='twilight')
            
            label = os.path.basename(file_path)
            ax.set_title(f'Phase Spectrogram: {label}')
            ax.set_ylabel('Frequency (Hz)')
            
            if i == num_files - 1:
                ax.set_xlabel('Time (s)')
            
            # Add colorbar
            plt.colorbar(img, ax=ax, label='Phase (rad)')
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Phase Spectrogram comparison saved to {output_filename}")

def compare_phase_features(file_list, output_filename="comparison_phase_features.png"):
    """
    Compares more interpretable phase features (Unwrapped Phase & Group Delay) 
    at a specific high-energy segment (transient).
    Group Delay is plotted separately for each file.
    """
    if not file_list:
        return

    # Just read the first file to find the peak (assuming synchronization or similar structure)
    try:
        if not os.path.exists(file_list[0]):
            return
        rate, data0 = wavfile.read(file_list[0])
        if len(data0.shape) > 1: data0 = data0[:, 0]
        
        # Find peak amplitude index to select a meaningful segment
        peak_idx = np.argmax(np.abs(data0))
        n_fft = 2048
        
        # Define segment range (centered on peak)
        start_idx = max(0, peak_idx - n_fft // 2)
        end_idx = start_idx + n_fft
        
    except Exception as e:
        print(f"Error determining segment: {e}")
        return

    num_files = len(file_list)
    # 1 plot for Phase (overlay) + num_files plots for Group Delay
    plt.figure(figsize=(12, 4 * (1 + num_files)))
    
    # Ax for Phase (Top)
    ax_phase = plt.subplot(num_files + 1, 1, 1)
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i, file_path in enumerate(file_list):
        if not os.path.exists(file_path):
            continue
            
        try:
            rate, data = wavfile.read(file_path)
            if len(data.shape) > 1: data = data[:, 0]
            
            if end_idx > len(data): 
                continue
                
            segment = data[start_idx:end_idx]
            
            # Apply Window
            window = np.hamming(len(segment))
            spectrum = np.fft.rfft(segment * window)
            freqs = np.fft.rfftfreq(len(segment), 1/rate)
            
            # 1. Unwrapped Phase
            phase = np.unwrap(np.angle(spectrum))
            
            # 2. Group Delay
            group_delay = -np.diff(phase)
            group_delay = np.append(group_delay, 0)
            
            label = os.path.basename(file_path)
            col = colors[i % len(colors)]
            
            # Plot Phase (Overlay)
            ax_phase.plot(freqs, phase, label=label, alpha=0.7, color=col)
            
            # Plot Group Delay (Separate)
            ax_gd = plt.subplot(num_files + 1, 1, i + 2)
            ax_gd.plot(freqs, group_delay, label=label, color=col)
            ax_gd.set_title(f'Group Delay: {label}')
            ax_gd.set_ylabel('Group Delay')
            ax_gd.set_xlabel('Frequency (Hz)')
            ax_gd.grid(True, alpha=0.3)
            ax_gd.set_xlim(0, 8000)
            # Zoom in Y a bit to handle potential wrapping spikes
            ax_gd.set_ylim(-3.5, 3.5) 
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    ax_phase.set_title(f'Unwrapped Phase Comparison (at t={start_idx/rate:.3f}s)')
    ax_phase.set_ylabel('Phase (radians)')
    ax_phase.set_xlabel('Frequency (Hz)')
    ax_phase.legend()
    ax_phase.grid(True, alpha=0.3)
    ax_phase.set_xlim(0, 8000)

    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Phase features saved to {output_filename}")

if __name__ == "__main__":
    # The specific files requested by the user
    files_to_compare = [
        "./F001_R1_E1_M1_BT.wav",
        "./F001_R1_E1_L1_BT.wav",
        "./F001_R1_E1_L2_BT.wav"
    ]
    
    # Check if files exist, if not try relative paths
    base_dir = "./"
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
        compare_phase_spectrograms(resolved_files, "F001_phase_spectrogram_comparison.png")
        compare_phase_features(resolved_files, "F001_phase_features_comparison.png")
