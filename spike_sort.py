import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import spikes

# MCS PyData tools
import McsPy
import McsPy.McsData

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

# VISUALIZATION TOOLS
import matplotlib.pyplot as plt

# SUPRESS WARNINGS
import warnings
warnings.filterwarnings('ignore')

def process_channel(file, channel_id):
    file = McsPy.McsData.RawData(file)
    electrode_stream = file.recordings[0].analog_streams[1]
    
    # Plot the entire analog stream channel
    spikes.plot_analog_stream_channel(electrode_stream, channel_id, from_in_s=0, to_in_s=None, show=True)
    
    info = electrode_stream.channel_infos[channel_id].info
    print("Bandwidth: %s - %s Hz" % (info['HighPassFilterCutOffFrequency'], info['LowPassFilterCutOffFrequency']))

    signal = electrode_stream.get_channel_in_range(channel_id, 0, electrode_stream.channel_data.shape[1])[0]

    noise_std = np.std(signal)
    noise_mad = np.median(np.absolute(signal)) / 0.6745
    print('Noise Estimate by Standard Deviation: {0:g} V'.format(noise_std))
    print('Noise Estimate by MAD Estimator     : {0:g} V'.format(noise_mad))

    spike_threshold = -5 * noise_mad # roughly -30 µV
    
    # Redraw the entire signal with the threshold line
    spikes.plot_analog_stream_channel(electrode_stream, channel_id, from_in_s=0, to_in_s=None, show=False)
    fs = int(electrode_stream.channel_infos[channel_id].sampling_frequency.magnitude)
    # Convert threshold to µV and plot it as an orange solid line
    threshold_line_y = spike_threshold * 1e6
    plt.axhline(y=threshold_line_y, color='orange', linestyle='-', label=f'Threshold: {threshold_line_y:.2f} µV')
    plt.legend()
    plt.show()
    
    # Detect threshold crossings
    crossings = spikes.detect_threshold_crossings(signal, fs, spike_threshold, 0.003) # dead time of 3 ms
    spks = spikes.align_to_minimum(signal, fs, crossings, 0.002) # search range 2 ms

    timestamps = spks / fs

    # Plot the entire signal and mark the detected spikes
    spikes.plot_analog_stream_channel(electrode_stream, channel_id, from_in_s=0, to_in_s=None, show=False)
    plt.plot(timestamps, [spike_threshold * 1e6] * len(timestamps), 'ro', ms=2, label='Detected Spikes')
    plt.legend()
    plt.show()

    pre = 0.001 # 1 ms
    post = 0.002 # 2 ms
    cutouts = spikes.extract_waveforms(signal, fs, spks, pre, post)
    print("Cutout array shape: " + str(cutouts.shape)) # number of spikes x number of samples

    # Plot all spike cutouts
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
        spikes.plot_waveforms(cutouts[idx,:], fs, pre, post, n=None, color=color, show=False)
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
