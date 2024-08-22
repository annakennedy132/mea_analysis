import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import McsPy
import McsPy.McsData

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
import pandas as pd
from scipy.fftpack import fft

import warnings
warnings.filterwarnings('ignore')

def extract_waveforms(signal, fs, spike_times, pre, post):
    cutouts = []
    pre_idx = int(pre * fs)
    post_idx = int(post * fs)
    
    spike_times = np.ravel(spike_times)
    spikes_idx = (spike_times * fs).astype(int)
    
    for index in spikes_idx:
        if index - pre_idx >= 0 and index + post_idx <= signal.shape[0]:
            cutout = signal[(index - pre_idx):(index + post_idx)]
            cutouts.append(cutout)
    
    return np.stack(cutouts) if cutouts else np.array([])

def plot_waveforms(cutouts, fs, pre, post, n=100, color='k', show=True):
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
    electrode_stream = file.recordings[0].analog_streams[1]
    timestamp_stream = file.recordings[0].timestamp_streams[0]

    if channel_id in timestamp_stream.timestamp_entity:
        timestamp_entity = timestamp_stream.timestamp_entity[channel_id]
        timestamps, unit = timestamp_entity.get_timestamps()
        spks = timestamps * unit.to(ureg.s).magnitude

        signal = electrode_stream.get_channel_in_range(channel_id, 0, electrode_stream.channel_data.shape[1])[0]

        # Parameters
        pre = 0.001  # 1 ms
        post = 0.002  # 2 ms
        fs = int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude)

        # Extract the waveforms
        cutouts = extract_waveforms(signal, fs, spks, pre, post)

        if cutouts.size == 0:
            print(f"No cutouts found for channel ID {channel_id}.")
            return
        
        print(f"Cutout array shape for channel ID {channel_id}: {cutouts.shape}")

        plot_waveforms(cutouts, fs, pre, post, n=None)

        min_amplitude = np.amin(cutouts, axis=1)
        max_amplitude = np.amax(cutouts, axis=1)

        _ = plt.figure(figsize=(8,8))
        _ = plt.plot(min_amplitude*1e6, max_amplitude*1e6, '.')
        _ = plt.xlabel('Min. Amplitude (%s)' % ureg.uV)
        _ = plt.ylabel('Max. Amplitude (%s)' % ureg.uV)
        _ = plt.title(f'Min/Max Spike Amplitudes for Channel {channel_id}')
        plt.show()

        scaler = StandardScaler()
        scaled_cutouts = scaler.fit_transform(cutouts)

        pca = PCA()
        pca.fit(scaled_cutouts)
        print(f'Explained variance ratio for Channel {channel_id}:', pca.explained_variance_ratio_)

        pca.n_components = 2
        transformed = pca.fit_transform(scaled_cutouts)

        _ = plt.figure(figsize=(8,8))
        _ = plt.plot(transformed[:,0], transformed[:,1], '.')
        _ = plt.xlabel('Principal Component 1')
        _ = plt.ylabel('Principal Component 2')
        _ = plt.title(f'PC1 vs PC2 for Channel {channel_id}')
        plt.show()

        n_components = int(input(f"Enter the number of GMM components for Channel {channel_id}: "))
        gmm = GaussianMixture(n_components=n_components, n_init=10)
        labels = gmm.fit_predict(transformed)

        _ = plt.figure(figsize=(8,8))
        for i in range(n_components):
            idx = labels == i
            _ = plt.plot(transformed[idx,0], transformed[idx,1], '.')
            _ = plt.title(f'Cluster assignments by a GMM for Channel {channel_id}')
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

    else:
        print(f"Channel ID {channel_id} not recorded.")

# Load data file
file_path = input("Input file path: ")
file = McsPy.McsData.RawData(file_path)

while True:
    try:
        channel_id = int(input("Enter the channel ID (or type -1 to quit): "))
        if channel_id == -1:
            break
        process_channel(file, channel_id)
    except ValueError:
        print("Invalid input. Please enter a valid integer for the channel ID.")
