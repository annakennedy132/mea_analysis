import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
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
                # Read the CSV, including power standard deviation (Power STD)
                data = pd.read_csv(csv_file_path, header=None, names=["Frequency", "Firing Rate", "Power", "Power STD"])
                
                if frequency_labels is None:
                    frequency_labels = data["Frequency"]

                # Handle missing 'Power' and 'Power STD' data for Baseline
                data.loc[data['Frequency'].str.contains('Baseline', case=False), ['Power', 'Power STD']] = pd.NA
                
                # Rename columns to include ND level
                data = data.rename(columns={
                    "Firing Rate": f"{inner_folder_name}_firing_rate",
                    "Power": f"{inner_folder_name}_power",
                    "Power STD": f"{inner_folder_name}_power_std"  # Include standard deviation
                })
            
                # Append data (firing rate, power, power_std) for this light level
                all_data.append(data[[f"{inner_folder_name}_firing_rate", f"{inner_folder_name}_power", f"{inner_folder_name}_power_std"]])
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

        # Separate columns into groups for each ND level (based on firing rate, power, and power_std)
        nd_levels = {}
        for col in self.combined_data.columns:
            if '_firing_rate' in col or '_power' in col or '_power_std' in col:
                nd_level = col.split('_')[0]  # Extract ND0, ND1, etc.
                if nd_level not in nd_levels:
                    nd_levels[nd_level] = {'firing_rate': None, 'power': None, 'power_std': None}
                if '_firing_rate' in col:
                    nd_levels[nd_level]['firing_rate'] = self.combined_data[col].values
                elif '_power' in col:
                    nd_levels[nd_level]['power'] = self.combined_data[col].values
                elif '_power_std' in col:
                    nd_levels[nd_level]['power_std'] = self.combined_data[col].values

        trial_folder_name = os.path.basename(self.trial_folder)
        output_pdf = os.path.join(self.trial_folder, f'collated_data_{trial_folder_name}.pdf')

        with PdfPages(output_pdf) as pdf:
            self._plot_power(frequency, nd_levels, pdf)  # Plot power with error bars
            self._plot_firing_rate(frequency, nd_levels, pdf)  # Plot firing rate without error bars

    def _plot_power(self, frequency, nd_levels, pdf):
        fig, ax = plt.subplots(figsize=(10, 6))
        shades_of_blue = sns.color_palette("Blues", len(nd_levels))

        # Skip the first frequency value (Baseline)
        plotting_frequencies = [1, 2, 5, 10, 20, 30]

        for i, (nd, vals) in enumerate(nd_levels.items()):
            if vals['power'] is not None:
                ax.plot(plotting_frequencies, vals['power'][1:], label=nd, color=shades_of_blue[i], linewidth=2, marker='o')
            if vals['power'] is not None and vals['power_std'] is not None:
            # Then, plot the error bars separately (without re-plotting the points)
                ax.errorbar(plotting_frequencies, vals['power'][1:], yerr=vals['power_std'][1:], 
                        fmt='none', ecolor=shades_of_blue[i], elinewidth=2, capsize=5)


        ax.set_xticks(plotting_frequencies)
        ax.set_xticklabels(frequency[1:], rotation=45)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(bottom=0)

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_title('Power')

        ax.legend(title='Light Levels')
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)

    def _plot_firing_rate(self, frequency, nd_levels, pdf):
        fig, ax = plt.subplots(figsize=(10, 6))
        shades_of_red = sns.color_palette("Reds", len(nd_levels))

        # Match the number of frequencies (including baseline) for firing rate
        plotting_frequencies = [0, 1, 2, 5, 10, 20, 30]

        for i, (nd, vals) in enumerate(nd_levels.items()):
            # Ensure firing rate values exist
            if vals['firing_rate'] is not None:
                ax.plot(plotting_frequencies, vals['firing_rate'], label=nd, color=shades_of_red[i], linewidth=2, marker='o')

        # Check if any lines were plotted, otherwise give a warning
        if len(ax.get_lines()) == 0:
            print("No valid data found to plot for firing rate.")

        ax.set_xticks(plotting_frequencies)
        ax.set_xticklabels(frequency[:len(plotting_frequencies)], rotation=45)  # Adjusted to match tick locations

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Firing Rate')
        ax.set_title('Firing Rate')

        ax.legend(title='Light Levels')
        plt.tight_layout()
        pdf.savefig()
        plt.close(fig)
