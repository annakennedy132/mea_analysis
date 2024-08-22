import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# MCS PyData tools
import McsPy
import McsPy.McsData

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

# VISUALIZATION TOOLS
import matplotlib.pyplot as plt
import pandas as pd

# SUPRESS WARNINGS
import warnings
warnings.filterwarnings('ignore')

def plot_analog_stream_channel(analog_stream, channel_idx, from_in_s=0, to_in_s=None, show=True):
    """
    Plots data from a single AnalogStream channel
    
    :param analog_stream: A AnalogStream object
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude
   
    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(to_in_s * sampling_frequency))
        
    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second
    
    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV

    # construct the plot
    _ = plt.figure(figsize=(20,6))
    _ = plt.plot(time_in_sec, signal_in_uV)
    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Channel %s' % channel_info.info['Label'])
    if show:
        plt.show()

def detect_threshold_crossings(signal, fs, threshold, dead_time):
    """
    Detect threshold crossings in a signal with dead time and return them as an array
    
    The signal transitions from a sample above the threshold to a sample below the threshold for a detection and
    the last detection has to be more than dead_time apart from the current one.
    
    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold: The threshold for the signal
    :param dead_time: The dead time in seconds. 
    """
    dead_time_idx = dead_time * fs
    threshold_crossings = np.diff((signal <= threshold).astype(int) > 0).nonzero()[0]
    distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
        threshold_crossings = threshold_crossings[distance_sufficient]
        distance_sufficient = np.insert(np.diff(threshold_crossings) >= dead_time_idx, 0, True)
    return threshold_crossings

def get_next_minimum(signal, index, max_samples_to_search):
    """
    Returns the index of the next minimum in the signal after an index
    
    :param signal: The signal as a 1-dimensional numpy array
    :param index: The scalar index 
    :param max_samples_to_search: The number of samples to search for a minimum after the index
    """
    search_end_idx = min(index + max_samples_to_search, signal.shape[0])
    min_idx = np.argmin(signal[index:search_end_idx])
    return index + min_idx

def align_to_minimum(signal, fs, threshold_crossings, search_range):
    """
    Returns the index of the next negative spike peak for all threshold crossings
    
    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param threshold_crossings: The array of indices where the signal crossed the detection threshold
    :param search_range: The maximum duration in seconds to search for the minimum after each crossing
    """
    search_end = int(search_range*fs)
    aligned_spikes = [get_next_minimum(signal, t, search_end) for t in threshold_crossings]
    return np.array(aligned_spikes)

def extract_waveforms(signal, fs, spikes_idx, pre, post):
    """
    Extract spike waveforms as signal cutouts around each spike index as a spikes x samples numpy array
    
    :param signal: The signal as a 1-dimensional numpy array
    :param fs: The sampling frequency in Hz
    :param spikes_idx: The sample index of all spikes as a 1-dim numpy array
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    """
    cutouts = []
    pre_idx = int(pre * fs)
    post_idx = int(post * fs)
    for index in spikes_idx:
        if index-pre_idx >= 0 and index+post_idx <= signal.shape[0]:
            cutout = signal[(index-pre_idx):(index+post_idx)]
            cutouts.append(cutout)
    return np.stack(cutouts)

def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
    """
    Plot an overlay of spike cutouts
    
    :param cutouts: A spikes x samples array of cutouts
    :param fs: The sampling frequency in Hz
    :param pre: The duration of the cutout before the spike in seconds
    :param post: The duration of the cutout after the spike in seconds
    :param n: The number of cutouts to plot, or None to plot all. Default: 100
    :param color: The line color as a pyplot line/marker style. Default: 'k'=black
    :param show: Set this to False to disable showing the plot. Default: True
    """
    if n is None:
        n = cutouts.shape[0]
    n = min(n, cutouts.shape[0])
    time_in_us = np.arange(-pre*1000, post*1000, 1e3/fs)
    if show:
        _ = plt.figure(figsize=(12,6))
    
    for i in range(n):
        _ = plt.plot(time_in_us, cutouts[i,]*1e6, color, linewidth=1, alpha=0.3)
        _ = plt.xlabel('Time (%s)' % ureg.ms)
        _ = plt.ylabel('Voltage (%s)' % ureg.uV)
        _ = plt.title('Cutouts')
    
    if show:
        plt.show()

def process_channel(file, channel_id):
    file = McsPy.McsData.RawData(file)
    electrode_stream = file.recordings[0].analog_streams[1]
    
    # Plot the entire analog stream channel
    plot_analog_stream_channel(electrode_stream, channel_id, from_in_s=0, to_in_s=None, show=True)
    
    info = electrode_stream.channel_infos[channel_id].info
    print("Bandwidth: %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))

    signal = electrode_stream.get_channel_in_range(channel_id, 0, electrode_stream.channel_data.shape[1])[0]

    noise_std = np.std(signal)
    noise_mad = np.median(np.absolute(signal)) / 0.6745
    print('Noise Estimate by Standard Deviation: {0:g} V'.format(noise_std))
    print('Noise Estimate by MAD Estimator     : {0:g} V'.format(noise_mad))

    spike_threshold = -5 * noise_mad # roughly -30 µV
    
    # Redraw the entire signal with the threshold line
    plot_analog_stream_channel(electrode_stream, channel_id, from_in_s=0, to_in_s=None, show=False)
    fs = int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude)
    # Convert threshold to µV and plot it as an orange solid line
    threshold_line_y = spike_threshold * 1e6
    plt.axhline(y=threshold_line_y, color='orange', linestyle='-', label=f'Threshold: {threshold_line_y:.2f} µV')
    plt.legend()
    plt.show()
    
    # Detect threshold crossings
    crossings = detect_threshold_crossings(signal, fs, spike_threshold, 0.003) # dead time of 3 ms
    spks = align_to_minimum(signal, fs, crossings, 0.002) # search range 2 ms

    timestamps = spks / fs

    # Plot the entire signal and mark the detected spikes
    plot_analog_stream_channel(electrode_stream, channel_id, from_in_s=0, to_in_s=None, show=False)
    plt.plot(timestamps, [spike_threshold * 1e6] * len(timestamps), 'ro', ms=2, label='Detected Spikes')
    plt.legend()
    plt.show()

    pre = 0.001 # 1 ms
    post = 0.002 # 2 ms
    cutouts = extract_waveforms(signal, fs, spks, pre, post)
    print("Cutout array shape: " + str(cutouts.shape)) # number of spikes x number of samples

    # Plot all spike cutouts
    plot_waveforms(cutouts, fs, pre, post, n=None)

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

    pca.n_components = 3
    transformed_3d = pca.fit_transform(scaled_cutouts)

    _ = plt.figure(figsize=(15,5))
    _ = plt.subplot(1, 3, 1)
    _ = plt.plot(transformed_3d[:,0], transformed_3d[:,1], '.')
    _ = plt.xlabel('Principal Component 1')
    _ = plt.ylabel('Principal Component 2')
    _ = plt.title('PC1 vs PC2')
    _ = plt.subplot(1, 3, 2)
    _ = plt.plot(transformed_3d[:,0], transformed_3d[:,2], '.')
    _ = plt.xlabel('Principal Component 1')
    _ = plt.ylabel('Principal Component 3')
    _ = plt.title('PC1 vs PC3')
    _ = plt.subplot(1, 3, 3)
    _ = plt.plot(transformed_3d[:,1], transformed_3d[:,2], '.')
    _ = plt.xlabel('Principal Component 2')
    _ = plt.ylabel('Principal Component 3')
    _ = plt.title('PC2 vs PC3')
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
        plot_waveforms(cutouts[idx,:], fs, pre, post, n=None, color=color, show=False)
    plt.show()

file = input("Input file path: ")

while True:
    try:
        channel_id = int(input("Enter the channel ID (or type -1 to quit): "))
        if channel_id == -1:
            break
        process_channel(file, channel_id)
    except ValueError:
        print("Invalid input. Please enter a valid integer for the channel ID.")
