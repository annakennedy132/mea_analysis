import csv
from matplotlib.backends.backend_pdf import PdfPages

def create_csv(list, filename):

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        for row in list:
            if row[1] is None:
                row = [row[0], "None", "None"]
            writer.writerow(row)

def save_report(figs, base_path, title=None):
    if title:
        report_path = f"{base_path}_{title}.pdf"
    else:
        report_path = f"{base_path}.pdf"
    
    with PdfPages(report_path) as pdf:
        for fig in figs:
            pdf.savefig(fig)