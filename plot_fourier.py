import numpy as np
import McsPy
import McsPy.McsData
from pint import UnitRegistry
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

import warnings
warnings.filterwarnings('ignore')

# Initialize unit registry for compatibility with McsPy
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

def compute_fourier(spike_train, sampling_rate):
    n = len(spike_train)
    T = 1.0 / sampling_rate
    yf = fft(spike_train)
    xf = fftfreq(n, T)[:n//2]
    return xf, 2.0/n * np.abs(yf[:n//2])

file_path = input("Input file path: ")
file = McsPy.McsData.RawData(file_path)
timestamp_stream = file.recordings[0].timestamp_streams[0]
event_stream = file.recordings[0].event_streams[0]

# Event IDs for start and stop events
start_event_id = 3
stop_event_id = 4

# Extract timestamps for the start and stop events
start_event_entity = event_stream.event_entity[start_event_id]
stop_event_entity = event_stream.event_entity[stop_event_id]
start_timestamps, time_unit = start_event_entity.get_event_timestamps()
stop_timestamps, _ = stop_event_entity.get_event_timestamps()
durations = stop_timestamps - start_timestamps
print("Start Times (s):", start_timestamps)
print("End Times (s):", stop_timestamps)
print("Durations (s):", durations)

channel_ids = list(timestamp_stream.timestamp_entity.keys())
channel_labels = [f'Channel {ch}' for ch in channel_ids]
stimulus_frequencies = [1, 2, 5, 10, 20, 30]

filtered_spike_timestamps = {freq: {channel_id: [] for channel_id in channel_ids} for freq in stimulus_frequencies}

for start, stop, freq in zip(start_timestamps, stop_timestamps, stimulus_frequencies):
    stimulus_duration = stop - start  # duration in microseconds
    interval_start = start
    interval_stop = start + stimulus_duration

    # Filter spikes for each channel within the time window
    for channel_id in timestamp_stream.timestamp_entity:
            timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
            timestamps, unit = timestamp_entity.get_timestamps()
            spks = np.array(timestamps, dtype=float)

            # Filter spikes within the current frequency interval
            filtered_spikes = spks[(spks >= interval_start) & (spks <= interval_stop)]
            filtered_spike_timestamps[freq][channel_id].extend(filtered_spikes)  # Align to interval start

sampling_interval = 100e-6  # 100 microseconds
sampling_rate = 1 / sampling_interval
window_duration = 1
power = []

for freq in stimulus_frequencies:
    #plt.figure(figsize=(15, 10))  # Initialize a figure for each stimulus frequency
    summed_fts = None
    num_channels = len(filtered_spike_timestamps[freq])

    # Use the start and stop times to define a fixed time window
    interval_start = min(start_timestamps)
    interval_stop = max(stop_timestamps)
    window_duration_us = interval_stop - interval_start
    window_duration_s = window_duration_us / 1e6  # Convert to seconds

    # Calculate the number of samples based on this fixed window duration
    num_samples = int(window_duration_s * sampling_rate)
    
    for channel_id, spikes in filtered_spike_timestamps[freq].items():
        if len(spikes) == 0:
            continue

        # Create the spike train with the exact number of samples
        spike_train = np.zeros(num_samples, dtype=np.float32)

        # Align the spikes to the start of the window
        spike_indices = ((np.array(spikes) - interval_start) * sampling_rate / 1e6).astype(int)
        spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(spike_train))]
        spike_train[spike_indices] = 1
        
        # Compute Fourier Transform with no padding
        freqs, fts = compute_fourier(spike_train, sampling_rate)

        # Limit to frequencies up to 50 Hz
        freq_mask = freqs <= 50
        limited_freqs = freqs[freq_mask]
        limited_fts = fts[freq_mask]

        # Plot each channel's FFT limited to 50 Hz
        plt.plot(limited_freqs, limited_fts, alpha=0.5, label=f'Channel {channel_id}')

        # Accumulate FFT results for averaging
        if summed_fts is None:
            summed_fts = limited_fts
        else:
            summed_fts += limited_fts

    # Display the plot for all channels for the current frequency
    plt.xlim(0, 50)  # Limit the x-axis from 0 to 50 Hz
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Fourier Transform for Frequency {freq} Hz (All Channels)')
    plt.legend()
    plt.show()

    # plot the averaged FFT
    if summed_fts is not None:
        average_fts = summed_fts / num_channels
        freq_index = (np.abs(limited_freqs - freq)).argmin()
        peak_amplitude = average_fts[freq_index]
        power.append(peak_amplitude)
        print(f"The amplitude at the stimulus frequency {freq} Hz is {peak_amplitude}")

        plt.figure(figsize=(15, 10))
        plt.plot(limited_freqs, average_fts)
        plt.xlim(0, 50)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(f'Averaged Fourier Transform for Frequency {freq} Hz')
        plt.show()
