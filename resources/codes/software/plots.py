import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plotting App")
        self.root.geometry("700x600")

        # Button to upload data
        self.upload_btn = ttk.Button(root, text="Upload Dataset", command=self.upload_data)
        self.upload_btn.pack(pady=10)

        # Dropdown for Plot Type
        self.plot_types = ["scatter", "line", "bar"]
        self.plot_combo = ttk.Combobox(root, values=self.plot_types, 
                                       state="readonly", 
                                       justify="center")
        self.plot_combo.current(0)
        self.plot_combo.pack(pady=10)

        # Dropdown for Sample Datasets or User Data
        self.datasets = ["Select a Dataset", "tips", "titanic"]
        self.data_combo = ttk.Combobox(root, values=self.datasets, 
                                       state="readonly", 
                                       justify="center")
        self.data_combo.current(0)
        self.data_combo.bind("<<ComboboxSelected>>", self.update_columns)
        self.data_combo.pack(pady=10)

        # Dropdowns for Numeric and Categorical Columns
        self.numeric_combo = ttk.Combobox(root, state="readonly", justify="center")
        self.numeric_combo.pack(pady=10)

        self.categorical_combo = ttk.Combobox(root, state="readonly", justify="center")
        self.categorical_combo.pack(pady=10)

        # Plot Button
        self.plot_btn = ttk.Button(root, text="Generate Plot", command=self.generate_plot)
        self.plot_btn.pack(pady=20)

        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=1)

        self.data = None

    def upload_data(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                self.datasets.append("User Data")
                self.data_combo["values"] = self.datasets
                self.data_combo.current(len(self.datasets) - 1)
                self.update_columns()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load dataset. Error: {str(e)}")

    def update_columns(self, event=None):
        dataset_name = self.data_combo.get()
        
        if dataset_name == "tips" or dataset_name == "titanic":
            self.data = sns.load_dataset(dataset_name)

        if self.data is not None:
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            categorical_cols = self.data.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
            
            self.numeric_combo["values"] = numeric_cols
            self.categorical_combo["values"] = categorical_cols

            if numeric_cols:
                self.numeric_combo.current(0)
            if categorical_cols:
                self.categorical_combo.current(0)

    def generate_plot(self):
        plot_type = self.plot_combo.get()
        numeric_col = self.numeric_combo.get()
        categorical_col = self.categorical_combo.get()

        # Plotting
        fig, ax = plt.subplots(figsize=(5, 4))

        if plot_type == "scatter":
            sns.scatterplot(data=self.data, x=numeric_col, y=categorical_col, ax=ax)
        elif plot_type == "line":
            self.data.groupby(categorical_col).mean()[numeric_col].plot(kind='line', ax=ax)
        elif plot_type == "bar":
            sns.barplot(data=self.data, x=categorical_col, y=numeric_col, ax=ax)

        self.show_plot(fig)

    def show_plot(self, fig):
        # Clear previous canvas frame
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotApp(root)
    root.mainloop()
