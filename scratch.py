import pdfplumber
with pdfplumber.open("corpus/employer/b1.1-maintaining-high-professional-standards-oct-22-SUPERSEDED.pdf") as pdf:
    print(pdf.pages[0].extract_text())