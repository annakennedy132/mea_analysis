import os
import numpy as np
import McsPy
import McsPy.McsData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import warnings
from utils import spikes

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

# VISUALIZATION TOOLS
import matplotlib.pyplot as plt

# SUPRESS WARNINGS
import warnings
warnings.filterwarnings('ignore')

# Load the H5 file
file_path = input("Input file path: ")
file = McsPy.McsData.RawData(file_path)

# Use the electrode stream for the raw data and timestamp stream for spikes
electrode_stream = file.recordings[0].analog_streams[1]
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

# Frequencies for the first 6 stimuli
stimulus_frequencies = [1, 2, 5, 10, 20, 30]

# Get the channel ID for both raw data and raster plots
channel_id = int(input(f"Enter the channel ID to plot: "))

# Get the analog data for the selected channel
channel_data = electrode_stream.channel_data[channel_id]

# Get the spike timestamps for the selected channel
timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
spike_timestamps, _ = timestamp_entity.get_timestamps()
spike_timestamps = np.array(spike_timestamps, dtype=float)

# Get the sampling frequency from the electrode stream
sampling_frequency = electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude

# Get the directory of the input file and create the output PDF file path
output_directory = os.path.dirname(file_path)
title = os.path.basename(output_directory)
output_pdf_path = os.path.join(output_directory, f'{title}_channel_{channel_id}.pdf')

# Plot both the raw data and raster plots and save to PDF
with PdfPages(output_pdf_path) as pdf:

    ### Part 1: Plot the Raw Data (First 6 Stimuli)
    plt.figure(figsize=(8, 8))

    for i, (start, stop, freq) in enumerate(zip(start_timestamps[:6], stop_timestamps[:6], stimulus_frequencies)):
        # Convert the start and stop times from microseconds to sample indices
        start_sample = int(((start - 1e6) / 1e6) * sampling_frequency)  # Include 1s before stimulus
        stop_sample = int(((stop + 1e6) / 1e6) * sampling_frequency)    # Include 1s after stimulus

        # Extract the raw data segment
        raw_data_segment = channel_data[start_sample:stop_sample]

        # Create a subplot for this stimulus
        ax = plt.subplot(6, 1, i + 1)

        # Plot the raw data
        time_axis = np.linspace(-1, (stop - start) / 1e6 + 1, len(raw_data_segment))  # Time in seconds from -1 to 11
        ax.plot(time_axis, raw_data_segment, color='slategrey')

        # Remove axis lines and ticks, except for the last plot
        if i < 5:  # Hide axis for all plots except the last one
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (s)')

        ax.set_yticks([])

        # Remove the x and y axis lines (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Label frequency on the left side of each subplot
        ax.text(-0.05, 0.5, f'{freq} Hz', transform=ax.transAxes, va='center', ha='right', fontsize=10)

    plt.tight_layout()
    pdf.savefig()  # Save the figure to the PDF
    plt.close()

    ### Part 2: Plot the Raster Plots (Spike Timestamps for the First 6 Stimuli)
    plt.figure(figsize=(8, 8))

    for i, (start, stop, freq) in enumerate(zip(start_timestamps[:6], stop_timestamps[:6], stimulus_frequencies)):
        # Filter spike timestamps between the start and stop times
        spikes_in_window = spike_timestamps[(spike_timestamps >= start) & (spike_timestamps <= stop)]

        # Convert timestamps from microseconds to seconds and adjust them relative to start time
        spikes_in_window = (spikes_in_window - start) / 1e6  # convert to seconds

        # Create a subplot for this stimulus
        ax = plt.subplot(6, 1, i + 1)

        # Compute spike density
        num_bins = 1000  # Number of bins for histogram
        hist, bin_edges = np.histogram(spikes_in_window, bins=num_bins, range=(0, (stop - start) / 1e6))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate the center of each bin

        # Use colormap to color the raster plot based on spike density
        cmap_name = "PuRd"
        cmap = get_cmap(cmap_name)
        norm = Normalize(vmin=0, vmax=np.max(hist))
        colors = cmap(norm(hist))

        for j in range(len(hist) - 1):  # Adjust loop range to avoid out-of-bounds
            ax.fill_between([bin_centers[j], bin_centers[j + 1]], 0, 1, color=colors[j], alpha=0.8)

        # Remove x and y ticks and labels for all except the last subplot
        if i < 5:  # Hide axis for all plots except the last one
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (s)')

        ax.set_yticks([])

        # Remove the x and y axis lines (spines)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Label frequency on the left side of each subplot
        ax.text(-0.05, 0.5, f'{freq} Hz', transform=ax.transAxes, va='center', ha='right', fontsize=10)

    plt.tight_layout()
    pdf.savefig()  # Save the figure to the PDF
    plt.close()

    ### Part 3: Plot sorted spikes
    info = electrode_stream.channel_infos[channel_id].info
    signal = electrode_stream.get_channel_in_range(channel_id, 0, electrode_stream.channel_data.shape[1])[0]
    noise_std = np.std(signal)
    noise_mad = np.median(np.absolute(signal)) / 0.6745
    spike_threshold = -5 * noise_mad # roughly -30 µV
    fs = int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude)
    crossings = spikes.detect_threshold_crossings(signal, fs, spike_threshold, 0.003)
    spks = spikes.align_to_minimum(signal, fs, crossings, 0.002)
    pre = 0.001 # 1 ms
    post = 0.002 # 2 ms
    cutouts = spikes.extract_waveforms(signal, fs, spks, pre, post)
    spikes.plot_waveforms(cutouts, fs, pre, post, n=None)

    min_amplitude = np.amin(cutouts, axis=1)
    max_amplitude = np.amax(cutouts, axis=1)

    _ = plt.figure(figsize=(8,8))
    _ = plt.plot(min_amplitude * 1e6, max_amplitude * 1e6, '.')
    _ = plt.xlabel('Min. Amplitude (%s)' % ureg.uV)
    _ = plt.ylabel('Max. Amplitude (%s)' % ureg.uV)
    _ = plt.title('Min/Max Spike Amplitudes')
    plt.show()

    scaler = StandardScaler()
    scaled_cutouts = scaler.fit_transform(cutouts)

    pca = PCA()
    pca.fit(scaled_cutouts)
    print(pca.explained_variance_ratio_)

    pca.n_components = 2
    transformed = pca.fit_transform(scaled_cutouts)

    _ = plt.figure(figsize=(8,8))
    _ = plt.plot(transformed[:,0], transformed[:,1], '.')
    _ = plt.xlabel('Principal Component 1')
    _ = plt.ylabel('Principal Component 2')
    _ = plt.title('PC1 vs PC2')
    plt.show()

    n_components = int(input("Input n_components: "))
    gmm = GaussianMixture(n_components=n_components, n_init=10)
    labels = gmm.fit_predict(transformed)

    _ = plt.figure(figsize=(8,8))
    for i in range(n_components):
        idx = labels == i
        _ = plt.plot(transformed[idx,0], transformed[idx,1], '.')
        _ = plt.title('Cluster assignments by a GMM')
        _ = plt.xlabel('Principal Component 1')
        _ = plt.ylabel('Principal Component 2')
        _ = plt.axis('tight')
    plt.show()

    _ = plt.figure(figsize=(8,8))
    for i in range(n_components):
        idx = labels == i
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][i]
        spikes.plot_waveforms(cutouts[idx,:], fs, pre, post, n=None, color=color, show=False)
    
    pdf.savefig()
    plt.close() 

print(f"Combined raw data and raster plot saved to {output_pdf_path}.")