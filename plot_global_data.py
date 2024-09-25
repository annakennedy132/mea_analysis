import os
import numpy as np
import McsPy
import McsPy.McsData
from pint import UnitRegistry
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import warnings
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

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

frequency_repeats = 3
stimulus_frequencies = [1, 2, 5, 10, 20, 30] * frequency_repeats
unique_frequencies = [1, 2, 5, 10, 20, 30]

channel_ids = list(timestamp_stream.timestamp_entity.keys())
channel_labels = [f'Channel {ch}' for ch in channel_ids]

# Extract and stitch together spike timestamps for each frequency
filtered_spike_timestamps = {freq: {channel_id: [] for channel_id in channel_ids} for freq in unique_frequencies}

# Iterate over each start and stop interval in the repeated sequence
for i, (start, stop, freq) in enumerate(zip(start_timestamps, stop_timestamps, stimulus_frequencies)):
    stimulus_duration = stop - start
    interval_start = start
    interval_stop = start + stimulus_duration

    # Calculate the correct cycle offset
    cycle_number = i // len(unique_frequencies)
    time_offset = cycle_number * stimulus_duration

    # Filter spikes for each selected channel within the time window
    for channel_id in channel_ids:
        timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
        timestamps, _ = timestamp_entity.get_timestamps()
        spks = np.array(timestamps, dtype=float)

        # Filter spikes within the current frequency interval and apply the time offset
        filtered_spikes = spks[(spks >= interval_start) & (spks <= interval_stop)]
        adjusted_spikes = filtered_spikes - interval_start + time_offset
        filtered_spike_timestamps[freq][channel_id].extend(adjusted_spikes)

# Get the directory of the input file
output_directory = os.path.dirname(file_path)
title = os.path.basename(output_directory)
output_pdf_path = os.path.join(output_directory, f'{title}_plots.pdf')

# Initialize a variable to store the active channels from the first frequency
first_active_channels = None

with PdfPages(output_pdf_path) as pdf:
    
    ### Part 1: Raster Plot Analysis
    for freq_idx, (freq, spikes_for_freq) in enumerate(filtered_spike_timestamps.items()):
        plt.figure(figsize=(12, 8), dpi=150)

        # Compute global maximum spike density across all channels for normalization
        all_hist = []
        max_spike_time = 0
        bin_width_us = 50000
        cmap_name = "PuRd"
        cmap = get_cmap(cmap_name)

        # Find the maximum spike time across channels for this frequency
        for spikes in spikes_for_freq.values():
            if len(spikes) > 200 and len(spikes) < 2000:
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

        # If this is the first frequency, store the active channels
        if freq_idx == 0:
            # Extract active channels based on spike count conditions
            first_active_channels = {ch: spikes for ch, spikes in spikes_for_freq.items() if len(spikes) > 10 and len(spikes) < 2000}

        # For all frequencies, use the active channels from the first frequency
        active_channels = {ch: spikes_for_freq[ch] for ch in first_active_channels.keys() if ch in spikes_for_freq}

        # Plot the raster plot for this frequency
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
        pdf.savefig()  # Save the figure to the PDF
        plt.close()


    ### Part 2: Fourier Transform Analysis

    sampling_interval_us = 100  # 100 microseconds
    sampling_rate = 1 / (sampling_interval_us * 1e-6)  # Convert microseconds to seconds
    window_duration_us = max(stop_timestamps) - min(start_timestamps)
    num_samples = int(window_duration_us / sampling_interval_us)

    power = []

    freq_range = 1  # ±0.5 Hz range around each target frequency

    for freq in unique_frequencies:
        summed_fts = None
        num_channels = len(filtered_spike_timestamps[freq])
        plt.figure(figsize=(12, 8))

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
        pdf.savefig()  # Save the figure to the PDF
        plt.close()

        # Plot the averaged FFT
        if summed_fts is not None:
            average_fts = summed_fts / num_channels

            # Find the maximum amplitude within the specified range around the target frequency
            indices_in_range = np.where((limited_freqs >= freq - freq_range) & (limited_freqs <= freq + freq_range))[0]
            if indices_in_range.size > 0:
                amplitudes_in_range = average_fts[indices_in_range]
                peak_amplitude = np.max(amplitudes_in_range)
            else:
                peak_amplitude = 0  # Default to 0 if no frequencies are in range

            power.append(peak_amplitude)
            print(f"The maximum amplitude within the range {freq - freq_range} to {freq + freq_range} Hz for the stimulus frequency {freq} Hz is {peak_amplitude}")

            plt.figure(figsize=(12, 8))
            plt.plot(limited_freqs, average_fts)
            plt.xlim(0, 50)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'Averaged Fourier Transform for Stimulus Frequency {freq} Hz')
            pdf.savefig()  # Save the figure to the PDF
            plt.close()

    ### Part 3: Scatter Plot of Amplitudes at Each Frequency

    plt.figure(figsize=(7,4))
    plt.scatter(unique_frequencies, power, color='darkblue')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Scatter Plot of Amplitudes at Each Stimulus Frequency')
    pdf.savefig()  # Save the scatter plot to the PDF
    plt.close()

    ### Part 4: Bar Graph Comparing Firing Rates During Stimuli vs Baseline

    # Calculate baseline firing rate for each channel
    baseline_rates_per_channel = []
    for channel_id in channel_ids:
        baseline_spikes = []
        baseline_total_duration = 0
        
        for i in range(1, len(start_timestamps)):
            baseline_start = stop_timestamps[i - 1]
            baseline_end = start_timestamps[i]
            
            if baseline_end > baseline_start:
                baseline_duration = baseline_end - baseline_start
                baseline_total_duration += baseline_duration
                
                timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
                timestamps, _ = timestamp_entity.get_timestamps()
                spks = np.array(timestamps, dtype=float)
                baseline_spikes.extend(spks[(spks >= baseline_start) & (spks <= baseline_end)])

        # Convert total baseline duration from microseconds to seconds
        baseline_total_duration_sec = baseline_total_duration / 1e6
        baseline_rate = len(baseline_spikes) / baseline_total_duration_sec if baseline_total_duration_sec > 0 else 0
        baseline_rates_per_channel.append(baseline_rate)

    # Calculate firing rates during each stimulus frequency for each channel
    firing_rates_per_channel = {freq: [] for freq in unique_frequencies}
    for freq in unique_frequencies:
        for channel_id in channel_ids:
            spikes = filtered_spike_timestamps[freq][channel_id]
            
            # The duration is calculated for each frequency by considering the stimulus time across all cycles
            num_cycles = stimulus_frequencies.count(freq)
            stimulus_duration_per_cycle = (stop_timestamps[0] - start_timestamps[0]) / 1e6  # Convert to seconds
            total_duration = num_cycles * stimulus_duration_per_cycle  # Total duration in seconds
            
            firing_rate = len(spikes) / total_duration if total_duration > 0 else 0
            firing_rates_per_channel[freq].append(firing_rate)

    # Combine firing rates with the baseline rates for the plot
    all_firing_rates = [np.mean(baseline_rates_per_channel)] + [np.mean(firing_rates_per_channel[freq]) for freq in unique_frequencies]
    x_labels = ["Baseline"] + [f"{freq} Hz" for freq in unique_frequencies]

    # Plot the bar graph with scatter points for each channel
    plt.figure(figsize=(6,4))
    bars = plt.bar(x_labels, all_firing_rates, color='red', alpha=0.7)

    # Overlay scatter points representing individual channel firing rates
    for rate in baseline_rates_per_channel:
        plt.scatter(0, rate, color='firebrick', s=10, alpha=0.7)
    for i, freq in enumerate(unique_frequencies):
        for rate in firing_rates_per_channel[freq]:
            plt.scatter(i + 1, rate, color='firebrick', s=10, alpha=0.7)

    plt.xlabel('Stimulus Frequency')
    plt.ylabel('Firing Rate (Hz)')
    plt.title('Firing Rates')
    plt.legend()
    pdf.savefig()
    plt.close()

print(f"All plots saved to {output_pdf_path}.")
