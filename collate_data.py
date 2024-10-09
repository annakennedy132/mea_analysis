import os
from classes.collate_plot import CollateAndPlot

def run():
    input_folder = input("Enter the folder path: ")
    trial_folders = [folder for folder in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, folder))]

    for trial_folder in trial_folders:
        trial_path = os.path.join(input_folder, trial_folder)
        cp = CollateAndPlot(trial_path)

        collated_data = cp.collate_csv_data()

        if collated_data is not None:
            output_file = os.path.join(trial_path, f'collated_data_{trial_folder}.csv')
            collated_data.to_csv(output_file, index=False)
            print(f"Collated data saved for folder {trial_folder} at {output_file}")
            cp.plot_power_firing_rate()
        else:
            print(f"No data found to collate for folder {trial_folder}.")

if __name__ == "__main__":
    run()
