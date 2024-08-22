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

# Initialize unit registry for compatibility with McsPy
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

def compute_fourier(spike_train, sampling_rate):
    n = len(spike_train)
    yf = fft(spike_train)
    xf = fftfreq(n, 1 / sampling_rate)[:n // 2]
    return xf, 2.0 / n * np.abs(yf[:n // 2])

# Load the H5 file
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
start_timestamps, _ = start_event_entity.get_event_timestamps()
stop_timestamps, _ = stop_event_entity.get_event_timestamps()
durations = stop_timestamps - start_timestamps

print("Start Times (µs):", start_timestamps)
print("End Times (µs):", stop_timestamps)
print("Durations (µs):", durations)

frequency_repeats = int(input("Input number of frequency repeats: "))
stimulus_frequencies = [1, 2, 5, 10, 20, 30] * frequency_repeats
unique_frequencies = [1, 2, 5, 10, 20, 30]

channel_ids = list(timestamp_stream.timestamp_entity.keys())
channel_labels = [f'Channel {ch}' for ch in channel_ids]

# Allow user to select the range of channels to plot
channel_start = int(input("Enter the starting channel ID: "))
channel_end = int(input("Enter the ending channel ID: "))

# Filter the channels based on user selection
selected_channels = [ch for ch in channel_ids if channel_start <= ch <= channel_end]

# Extract and stitch together spike timestamps for each frequency
filtered_spike_timestamps = {freq: {channel_id: [] for channel_id in selected_channels} for freq in unique_frequencies}

# Iterate over each start and stop interval in the repeated sequence
for i, (start, stop, freq) in enumerate(zip(start_timestamps, stop_timestamps, stimulus_frequencies)):
    stimulus_duration = stop - start
    interval_start = start
    interval_stop = start + stimulus_duration

    # Calculate the correct cycle offset
    cycle_number = i // len(unique_frequencies)
    time_offset = cycle_number * stimulus_duration

    # Filter spikes for each selected channel within the time window
    for channel_id in selected_channels:
        timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
        timestamps, _ = timestamp_entity.get_timestamps()
        spks = np.array(timestamps, dtype=float)

        # Filter spikes within the current frequency interval and apply the time offset
        filtered_spikes = spks[(spks >= interval_start) & (spks <= interval_stop)]
        adjusted_spikes = filtered_spikes - interval_start + time_offset
        filtered_spike_timestamps[freq][channel_id].extend(adjusted_spikes)

### Part 1: Raster Plot Analysis

bin_width_us = int(input("Input bin width (in µs): "))
cmap_name = 'PuRd'
cmap = get_cmap(cmap_name)

for freq, spikes_for_freq in filtered_spike_timestamps.items():
    plt.figure(figsize=(12, 8), dpi=150)

    # Compute global maximum spike density across all channels for normalization
    all_hist = []
    max_spike_time = 0
    for spikes in spikes_for_freq.values():
        if len(spikes) > 0:
            max_spike_time = max(max_spike_time, np.max(spikes))
    if max_spike_time > 0:
        bin_edges = np.arange(0, max_spike_time + bin_width_us, bin_width_us)
        for spikes in spikes_for_freq.values():
            hist, _ = np.histogram(spikes, bins=bin_edges)
            all_hist.extend(hist)
        global_max_density = np.max(all_hist) if all_hist else 1
    else:
        global_max_density = 1

    # Set up the colormap normalization
    norm = Normalize(vmin=0, vmax=global_max_density)
    sm = ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])

    # Plot the raster plot for this frequency
    active_channels = {ch: spikes for ch, spikes in spikes_for_freq.items() if len(spikes) > 0}
    for ch_idx, (channel, spikes) in enumerate(active_channels.items()):
        if len(spikes) > 0:
            hist, bin_edges = np.histogram(spikes, bins=bin_edges)
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            colors = cmap(norm(hist))
            # Use plt.fill_between for better control over filling areas
            for t in range(len(bin_centers) - 1):
                plt.fill_between(
                    [bin_centers[t] / 1e6, bin_centers[t + 1] / 1e6],  # Convert time to seconds
                    [ch_idx] * 2, 
                    [ch_idx + 1] * 2, 
                    color=colors[t]
                )

    plt.xlabel('Time (s)')  # x-axis is now in seconds
    plt.ylabel('Channel')
    plt.title(f'Raster Plot for {freq} Hz')
    plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Spike Density')
    plt.yticks(ticks=np.arange(len(active_channels)), labels=[f'{ch}' for ch in active_channels.keys()], fontsize=4)  # Adjust fontsize
    plt.tight_layout()
    plt.show()

### Part 2: Fourier Transform Analysis

sampling_interval_us = 100  # 100 microseconds
sampling_rate = 1 / (sampling_interval_us * 1e-6)  # Convert microseconds to seconds
window_duration_us = max(stop_timestamps) - min(start_timestamps)
num_samples = int(window_duration_us / sampling_interval_us)

power = []

for freq in unique_frequencies:
    summed_fts = None
    num_channels = len(filtered_spike_timestamps[freq])

    for channel_id, spikes in filtered_spike_timestamps[freq].items():
        if len(spikes) == 0:
            continue

        # Create the spike train with the exact number of samples
        spike_train = np.zeros(num_samples, dtype=np.float32)

        # Align the spikes to the start of the window
        spike_indices = ((np.array(spikes) - min(start_timestamps)) / sampling_interval_us).astype(int)
        spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < num_samples)]
        spike_train[spike_indices] = 1
        
        # Compute Fourier Transform with no padding
        freqs, fts = compute_fourier(spike_train, sampling_rate)

        # Limit to frequencies up to 50 Hz
        freq_mask = freqs <= 50
        limited_freqs = freqs[freq_mask]
        limited_fts = fts[freq_mask]

        # Plot each channel's FFT limited to 50 Hz (no legend)
        plt.plot(limited_freqs, limited_fts, alpha=0.5)

        # Accumulate FFT results for averaging
        if summed_fts is None:
            summed_fts = limited_fts
        else:
            summed_fts += limited_fts

    # Display the plot for all channels for the current frequency (no legend)
    plt.xlim(0, 50)  # Limit the x-axis from 0 to 50 Hz
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Fourier Transform for Frequency {freq} Hz (All Channels)')
    plt.show()

    # Plot the averaged FFT
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
