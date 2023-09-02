# Importing the tkinter module for GUI development
import tkinter as tk
# Importing filedialog and messagebox from tkinter for opening file dialog and displaying message boxes
from tkinter import filedialog, messagebox
# Importing pandas for data manipulation and analysis
import pandas as pd
# Importing ydata_profiling for generating data profiling reports
import ydata_profiling
# Importing webbrowser for opening the generated report in the default web browser
import webbrowser
# Importing seaborn for loading sample data
import seaborn as sns

# Function to upload a data file
def upload_file():
    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename(title="Open Data File", filetypes=(("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")))
    # If a file is selected
    if file_path:
        try:
            # Read the data from the selected file using pandas
            data = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            # Generate a report for the data
            generate_report(data)
        except Exception as e:
            # Display an error message if the data cannot be loaded
            messagebox.showerror("Error", f"Failed to load data. Error: {e}")

# Function to load sample data
def load_sample_data():
    # Load the tips dataset from seaborn
    data = sns.load_dataset('tips')
    # Generate a report for the data
    generate_report(data)

# Function to generate a report for the data
def generate_report(data):
    try:
        # Display an info message
        messagebox.showinfo("Info", "Generating report. Please wait...")
        # Generate a data profiling report using ydata_profiling
        report = data.profile_report()
        # Save the report to an HTML file
        report.to_file('report.html')
        # Open the report in the default web browser
        webbrowser.open('report.html')
    except Exception as e:
        # Display an error message if the report cannot be generated
        messagebox.showerror("Error", f"Failed to generate report. Error: {e}")

# GUI Setup
# Create a new tkinter application
app = tk.Tk()
# Set the title of the application
app.title("Automatic EDA Tool")

# Create a new frame for the GUI
frame = tk.Frame(app)
# Add padding to the frame
frame.pack(padx=20, pady=20)

# Create a label for the GUI
label = tk.Label(frame, text="Automatic EDA with Pandas Profiling", font=("Arial", 16, "bold"))
# Add padding to the label
label.pack(pady=20)

# Create a button to upload a data file
upload_button = tk.Button(frame, text="Upload Data", command=upload_file, width=20, height=2)
# Add padding to the button
upload_button.pack(pady=10)

# Create a button to load sample data
sample_button = tk.Button(frame, text="Load Sample Data", command=load_sample_data, width=20, height=2)
# Add padding to the button
sample_button.pack(pady=10)

# Start the GUI application
app.mainloop()
