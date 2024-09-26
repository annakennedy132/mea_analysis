import os
import numpy as np
import McsPy
import McsPy.McsData
from pint import UnitRegistry
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap

from utils import fourier, files

class GlobalPlots:
    
    def __init__(self, file, save_figs=False):
        self.file_path = file
        self.timestamp_stream = self.file_path.recordings[0].timestamp_streams[0]
        self.event_stream = self.file_path.recordings[0].event_streams[0]

        frequency_repeats = 3
        self.stimulus_frequencies = [1, 2, 5, 10, 20, 30] * frequency_repeats
        self.unique_frequencies = [1, 2, 5, 10, 20, 30]

        self.channel_ids = list(self.timestamp_stream.timestamp_entity.keys())
        self.channel_labels = [f'Channel {ch}' for ch in self.channel_ids]

        output_directory = os.path.dirname(self.file_path)
        title = os.path.basename(output_directory)
        self.output_pdf_path = os.path.join(output_directory, f'{title}_plots.pdf')

        self.first_active_channels = None
        self.figs = []
        self.save_figs = save_figs

        sampling_interval_us = 100  # 100 microseconds
        self.sampling_rate = 1 / (sampling_interval_us * 1e-6)

        self.power = []
        self.freq_range = 0.5  # ±0.5 Hz range around each target frequency

    def get_timestamps(self):
        # Event IDs for start and stop events
        start_event_id = 3
        stop_event_id = 4
        start_event_entity = self.event_stream.event_entity[start_event_id]
        stop_event_entity = self.event_stream.event_entity[stop_event_id]
        start_timestamps, _ = start_event_entity.get_event_timestamps()
        stop_timestamps, _ = stop_event_entity.get_event_timestamps()
        durations = stop_timestamps - start_timestamps

        print("Start Times (µs):", start_timestamps)
        print("End Times (µs):", stop_timestamps)
        print("Durations (µs):", durations)

        # Extract and stitch together spike timestamps for each frequency
        self.filtered_spike_timestamps = {freq: {channel_id: [] for channel_id in self.channel_ids} for freq in self.unique_frequencies}

        # Iterate over each start and stop interval in the repeated sequence
        for i, (start, stop, freq) in enumerate(zip(start_timestamps, stop_timestamps, self.stimulus_frequencies)):

            stimulus_duration = stop - start
            interval_start = start
            interval_stop = start + stimulus_duration

            # Calculate the correct cycle offset
            cycle_number = i // len(self.unique_frequencies)
            time_offset = cycle_number * stimulus_duration

            # Filter spikes for each selected channel within the time window
            for channel_id in self.channel_ids:
                timestamp_entity = self.timestamp_stream.timestamp_entity[channel_id]
                timestamps, _ = timestamp_entity.get_timestamps()
                spks = np.array(timestamps, dtype=float)

                # Filter spikes within the current frequency interval and apply the time offset
                filtered_spikes = spks[(spks >= interval_start) & (spks <= interval_stop)]
                adjusted_spikes = filtered_spikes - interval_start + time_offset
                self.filtered_spike_timestamps[freq][channel_id].extend(adjusted_spikes)

    def plot_raster(self):
        for freq_idx, (freq, spikes_for_freq) in enumerate(self.filtered_spike_timestamps.items()):
            fig, ax = plt.subplots(figsize=(12, 8), dpi=150)  # Create figure object

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
                first_active_channels = {ch: spikes for ch, spikes in spikes_for_freq.items() if len(spikes) > 100 and len(spikes) < 2000}

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

            ax.set_xlabel('Time (s)')  # x-axis is now in seconds
            ax.set_ylabel('Channel')
            ax.set_title(f'Raster Plot for {freq} Hz')
            plt.colorbar(sm, ax=ax, orientation='vertical', label='Spike Density')
            ax.set_yticks(np.arange(len(active_channels)))  # Set y-axis ticks
            ax.set_yticklabels([f'{ch}' for ch in active_channels.keys()], fontsize=4)  # Adjust fontsize
            plt.tight_layout()

            # Append the figure to the list
            self.figs.append(fig)
            plt.close(fig)

    def plot_fourier(self):
        window_duration_us = max(self.stop_timestamps) - min(self.start_timestamps)
        num_samples = int(window_duration_us / self.sampling_interval_us)

        for freq in self.unique_frequencies:
            summed_fts = None
            num_channels = len(self.filtered_spike_timestamps[freq])
            fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes

            for channel_id, spikes in self.filtered_spike_timestamps[freq].items():
                if len(spikes) == 0:
                    continue

                # Create the spike train with the exact number of samples
                spike_train = np.zeros(num_samples, dtype=np.float32)

                # Align the spikes to the start of the window
                spike_indices = ((np.array(spikes) - min(self.start_timestamps)) / self.sampling_interval_us).astype(int)
                spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < num_samples)]
                spike_train[spike_indices] = 1

                # Compute Fourier Transform with no padding
                freqs, fts = fourier.compute_fourier(spike_train, self.sampling_rate)

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

            ax.set_xlim(0, 50)  # Limit the x-axis from 0 to 50 Hz
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Fourier Transform for Frequency {freq} Hz (All Channels)')
            
            # Append the figure to the list
            self.figs.append(fig)
            plt.close(fig)  # Close the figure to avoid displaying it during execution


            # Plot the averaged FFT
            if summed_fts is not None:
                average_fts = summed_fts / num_channels

                # Find the maximum amplitude within the specified range around the target frequency
                indices_in_range = np.where((limited_freqs >= freq - self.freq_range) & (limited_freqs <= freq + self.freq_range))[0]
                if indices_in_range.size > 0:
                    amplitudes_in_range = average_fts[indices_in_range]
                    peak_amplitude = np.max(amplitudes_in_range)
                else:
                    peak_amplitude = 0  # Default to 0 if no frequencies are in range

                self.power.append(peak_amplitude)
                print(f"The maximum amplitude within the range {freq - self.freq_range} to {freq + self.freq_range} Hz for the stimulus frequency {freq} Hz is {peak_amplitude}")

                fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes
                ax.plot(limited_freqs, average_fts)
                ax.set_xlim(0, 50)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'Averaged Fourier Transform for Stimulus Frequency {freq} Hz')
                
                self.figs.append(fig)
                plt.close(fig)
                
    def plot_power(self):
        fig, ax = plt.subplots(figsize=(7, 4))  # Create figure and axes
        ax.scatter(self.unique_frequencies, self.power, color='darkblue')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Scatter Plot of Amplitudes at Each Stimulus Frequency')
        self.figs.append(fig)
        plt.close(fig)

    def plot_firing_rate(self):
        baseline_rates_per_channel = []

        for channel_id in self.channel_ids:
            baseline_spikes = []
            baseline_total_duration = 0
            
            for i in range(1, len(self.start_timestamps)):
                baseline_start = self.stop_timestamps[i - 1]
                baseline_end = self.start_timestamps[i]
                
                if baseline_end > baseline_start:
                    baseline_duration = baseline_end - baseline_start
                    baseline_total_duration += baseline_duration
                    
                    timestamp_entity = self.timestamp_stream.timestamp_entity[channel_id]
                    timestamps, _ = timestamp_entity.get_timestamps()
                    spks = np.array(timestamps, dtype=float)
                    baseline_spikes.extend(spks[(spks >= baseline_start) & (spks <= baseline_end)])

            # Convert total baseline duration from microseconds to seconds
            baseline_total_duration_sec = baseline_total_duration / 1e6
            baseline_rate = len(baseline_spikes) / baseline_total_duration_sec if baseline_total_duration_sec > 0 else 0
            baseline_rates_per_channel.append(baseline_rate)

        # Calculate firing rates during each stimulus frequency for each channel
        firing_rates_per_channel = {freq: [] for freq in self.unique_frequencies}
        for freq in self.unique_frequencies:
            for channel_id in self.channel_ids:
                spikes = self.filtered_spike_timestamps[freq][channel_id]
                
                # The duration is calculated for each frequency by considering the stimulus time across all cycles
                num_cycles = self.stimulus_frequencies.count(freq)
                stimulus_duration_per_cycle = (self.stop_timestamps[0] - self.start_timestamps[0]) / 1e6  # Convert to seconds
                total_duration = num_cycles * stimulus_duration_per_cycle  # Total duration in seconds
                
                firing_rate = len(spikes) / total_duration if total_duration > 0 else 0
                firing_rates_per_channel[freq].append(firing_rate)

        # Combine firing rates with the baseline rates for the plot
        all_firing_rates = [np.mean(baseline_rates_per_channel)] + [np.mean(firing_rates_per_channel[freq]) for freq in self.unique_frequencies]
        x_labels = ["Baseline"] + [f"{freq} Hz" for freq in self.unique_frequencies]

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(x_labels, all_firing_rates, color='red', alpha=0.7)

        for rate in baseline_rates_per_channel:
            ax.scatter(0, rate, color='firebrick', s=10, alpha=0.7)
        for i, freq in enumerate(self.unique_frequencies):
            for rate in firing_rates_per_channel[freq]:
                ax.scatter(i + 1, rate, color='firebrick', s=10, alpha=0.7)

        ax.set_xlabel('Stimulus Frequency')
        ax.set_ylabel('Firing Rate (Hz)')
        ax.set_title('Firing Rates')
        ax.legend()
        self.figs.append(fig)
        plt.close(fig)

    def save_data_to_csv(self):
        # Prepare CSV output file path in the same directory as the PDF output
        output_directory = os.path.dirname(self.file_path)
        csv_file_path = os.path.join(output_directory, 'firing_rates_and_power.csv')

        # Prepare data for CSV: baseline firing rates, stimulus firing rates, and power
        baseline_rate_mean = np.mean(self.baseline_rates_per_channel)
        stimulus_rates_means = [np.mean(self.firing_rates_per_channel[freq]) for freq in self.unique_frequencies]

        # Create the rows for the CSV
        csv_data = [['Baseline', baseline_rate_mean, None]]  # Baseline row (no power for baseline)
        for freq, rate, power in zip(self.unique_frequencies, stimulus_rates_means, self.power):
            csv_data.append([f'{freq} Hz', rate, power])

        files.create_csv(csv_data, csv_file_path)



    