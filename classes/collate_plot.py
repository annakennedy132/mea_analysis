import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

class CollateAndPlot:
    def __init__(self, trial_folder):
        self.trial_folder = trial_folder
        self.combined_data = None

    def collate_csv_data(self):
        all_data = []
        frequency_labels = None

        # Loop through inner folders (e.g., ND0, ND1, etc.)
        for nd_folder in glob.glob(os.path.join(self.trial_folder, 'ND*')):
            inner_folder_name = os.path.basename(nd_folder)
            csv_file_path = os.path.join(nd_folder, 'firing_rates_and_power.csv')
            
            if os.path.exists(csv_file_path):
                data = pd.read_csv(csv_file_path, header=None, names=["Frequency", "Firing Rate", "Power"])
                
                if frequency_labels is None:
                    frequency_labels = data["Frequency"]

                # Handle missing 'Power' data for Baseline
                data.loc[data['Frequency'].str.contains('Baseline', case=False), 'Power'] = pd.NA
                
                data = data.rename(columns={
                    "Firing Rate": f"{inner_folder_name}_firing_rate",
                    "Power": f"{inner_folder_name}_power"
                })
            
                all_data.append(data[[f"{inner_folder_name}_firing_rate", f"{inner_folder_name}_power"]])
            else:
                print(f"CSV file not found in {inner_folder_name}")

        if all_data:
            combined_data = pd.concat(all_data, axis=1)
            combined_data.insert(0, 'Frequency', frequency_labels)
            self.combined_data = combined_data
            return combined_data
        else:
            print(f"No data found to collate for folder {self.trial_folder}.")
            return None

    def plot_power_firing_rate(self):
        if self.combined_data is None:
            print("No combined data available to plot.")
            return
        
        frequency = self.combined_data['Frequency']

        # Separate columns into groups for each ND level (based on firing rate and power)
        nd_levels = {}
        for col in self.combined_data.columns:
            if '_firing_rate' in col or '_power' in col:
                nd_level = col.split('_')[0]  # Extract ND0, ND1, etc.
                if nd_level not in nd_levels:
                    nd_levels[nd_level] = {'firing_rate': [], 'power': []}
                if '_firing_rate' in col:
                    nd_levels[nd_level]['firing_rate'].append(self.combined_data[col])
                elif '_power' in col:
                    nd_levels[nd_level]['power'].append(self.combined_data[col])

        # Calculate averages and standard deviation for each ND level
        avg_power = {nd: np.nanmean(np.stack(vals['power'], axis=1), axis=1) for nd, vals in nd_levels.items()}
        avg_firing_rate = {nd: np.nanmean(np.stack(vals['firing_rate'], axis=1), axis=1) for nd, vals in nd_levels.items()}

        trial_folder_name = os.path.basename(self.trial_folder)
        output_pdf = os.path.join(self.trial_folder, f'collated_data_{trial_folder_name}.pdf')

        with PdfPages(output_pdf) as pdf:
            self._plot_avg_power(frequency, avg_power, pdf)
            self._plot_avg_firing_rate(frequency, avg_firing_rate, pdf)

    def _plot_avg_power(self, frequency, avg_power, pdf):
        fig, ax = plt.subplots(figsize=(10, 6))
        shades_of_blue = sns.color_palette("Blues", len(avg_power))

        plotting_frequencies = [1, 2, 5, 10, 20 ,30]

        for i, (nd, avg_vals) in enumerate(avg_power.items()):
            ax.plot(plotting_frequencies, avg_vals[1:], label=nd, color=shades_of_blue[i], linewidth=2, marker='o')

        ax.set_xticks(plotting_frequencies)
        ax.set_xticklabels(frequency[1:], rotation=45)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Average Power')
        ax.set_title('Power Plot')

        ax.legend(title='Light Levels')
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)

    def _plot_avg_firing_rate(self, frequency, avg_firing_rate, pdf):
        fig, ax = plt.subplots(figsize=(10, 6))
        shades_of_red = sns.color_palette("Reds", len(avg_firing_rate))

        plotting_frequencies = [0, 1, 2, 5, 10, 20 ,30]

        for i, (nd, avg_vals) in enumerate(avg_firing_rate.items()):
            ax.plot(plotting_frequencies, avg_vals, label=nd, color=shades_of_red[i], linewidth=2, marker='o')

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Average Firing Rate')
        ax.set_title('Firing Rate')
        ax.legend(title='Light Levels')

        ax.set_xticks(plotting_frequencies)
        ax.set_xticklabels(frequency[1:], rotation=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)
