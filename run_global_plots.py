from classes.global_plots import GlobalPlots

def run():

    file_path = input("Input file path: ")
    experiment = input("Input experiment ID: ")
    group = input("input experiment group: ")
    light_level = input("Input light level: ")

    gp = GlobalPlots(file_path, experiment, group, light_level)
    gp.get_timestamps()
    gp.plot_raster()
    gp.plot_fourier()
    gp.plot_power()
    gp.plot_firing_rate()
    gp.save_data()
    gp.save_pdf_report()

if __name__ == "__main__":
    run()