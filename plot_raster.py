import os
import numpy as np
import McsPy
import McsPy.McsData

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

import warnings
warnings.filterwarnings('ignore')

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

# Iterate over each start and stop interval
for start, stop in zip(start_timestamps, stop_timestamps):
    stimulus_duration = stop - start
    for freq in stimulus_frequencies:
        # Define the interval for the current frequency
        interval_start = start
        interval_stop = start + stimulus_duration

        # Filter spikes for each channel within the time window
        for channel_id in timestamp_stream.timestamp_entity:
            timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
            timestamps, unit = timestamp_entity.get_timestamps()
            spks = np.array(timestamps, dtype=float)

            # Filter spikes within the current frequency interval
            filtered_spikes = spks[(spks >= interval_start) & (spks <= interval_stop)]
            filtered_spike_timestamps[freq][channel_id].extend(filtered_spikes - interval_start)  # Align to interval start

bin_width_sec = int(input("Input bin width: "))
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
        for spikes in spikes_for_freq.values():
            hist, _ = np.histogram(spikes, bins=np.arange(0, max_spike_time + bin_width_sec, bin_width_sec))
            all_hist.extend(hist)
        global_max_density = np.max(all_hist) if all_hist else 1  # Avoid division by zero
    else:
        global_max_density = 1

    # Set up the colormap normalization
    vmin = 0
    vmax = global_max_density
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])

    # Plot the raster plot for this frequency
    active_channels = {ch: spikes for ch, spikes in spikes_for_freq.items() if len(spikes) > 0}
    for ch_idx, (channel, spikes) in enumerate(active_channels.items()):
        if len(spikes) > 0:
            hist, bin_edges = np.histogram(spikes, bins=np.arange(0, max_spike_time + bin_width_sec, bin_width_sec))
            bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
            colors = cmap(norm(hist))
            # Use plt.fill_between for better control over filling areas
            for t in range(len(bin_centers) - 1):
                plt.fill_between([bin_centers[t], bin_centers[t + 1]], [ch_idx] * 2, [ch_idx + 1] * 2, color=colors[t])

    plt.xlabel('Time (s)')
    plt.ylabel('Channel')
    plt.title(f'Raster Plot for {freq} Hz')
    plt.colorbar(sm, ax=plt.gca(), orientation='vertical', label='Spike Density')
    plt.yticks(np.arange(len(active_channels)), [f'{ch}' for ch in active_channels.keys()], fontsize=4)  # Adjust fontsize
    plt.tight_layout()
    plt.show()
