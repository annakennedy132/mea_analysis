from classes.global_plots import GlobalPlots

def run():

    file_path = input("Input file path: ")
    gp = GlobalPlots(file_path)
    gp.get_timestamps()
    gp.plot_raster()
    gp.plot_fourier()
    gp.plot_power()
    gp.plot_firing_rate()
    gp.save_data()
    gp.save_pdf_report()

if __name__ == "__main__":
    run()