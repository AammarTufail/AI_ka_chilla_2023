import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import seaborn as sns
import webbrowser
from pandas_profiling import ProfileReport
class EDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Simple EDA App")
        self.root.geometry("400x250")

        # Label
        self.label = ttk.Label(root, text="Upload Dataset or Select Sample Data")
        self.label.pack(pady=20)

        # Upload Dataset Button
        self.upload_btn = ttk.Button(root, text="Upload Dataset", command=self.upload_file)
        self.upload_btn.pack(pady=10)

        # Dropdown for Sample Datasets
        self.datasets = ["Select Sample Dataset", "tips", "titanic"]
        self.combo = ttk.Combobox(root, values=self.datasets)
        self.combo.current(0)
        self.combo.pack(pady=10)

        # Button to Run EDA
        self.eda_btn = ttk.Button(root, text="Run EDA", command=self.run_eda)
        self.eda_btn.pack(pady=10)

        # Checkbox to Show in Browser
        self.browser_var = tk.IntVar()
        self.browser_check = ttk.Checkbutton(root, text="Open in Browser", variable=self.browser_var)
        self.browser_check.pack(pady=10)

        self.data = None

    def upload_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                messagebox.showinfo("Success", "Dataset Loaded Successfully!")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def select_sample_data(self):
        dataset_name = self.combo.get()
        if dataset_name == "Select Sample Dataset":
            self.data = None
            return
        try:
            self.data = sns.load_dataset(dataset_name)
            messagebox.showinfo("Success", f"{dataset_name} Dataset Loaded Successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def run_eda(self):
        if self.combo.get() != "Select Sample Dataset":
            self.select_sample_data()
        if self.data is None:
            messagebox.showerror("Error", "No Dataset Loaded")
            return
        try:
            report = ProfileReport(self.data)
            report.to_file("eda_report.html")
            messagebox.showinfo("Success", "EDA Report Generated Successfully!")
            if self.browser_var.get():
                webbrowser.open("eda_report.html", new=2)
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = EDAApp(root)
    root.mainloop()
