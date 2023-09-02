import tkinter as tk
from tkcalendar import DateEntry
from datetime import datetime

def calculate_age():
    try:
        birth_date = birth_date_entry.get_date()
        end_date = end_date_entry.get_date()
        
        difference = end_date - birth_date
        seconds = difference.total_seconds()
        minutes = seconds // 60
        hours = minutes // 60
        days = hours // 24
        weeks = days // 7
        months = days // 30.44  # Average number of days in a month
        years = days // 365.25  # Average number of days in a year
        
        result_text = f"""
You have lived for:
{seconds} seconds,
{minutes} minutes,
{hours} hours,
{days} days,
{weeks} weeks,
{months:.2f} months,
{years:.2f} years.
"""
        label_result.config(text=result_text)
    except ValueError:
        label_result.config(text="Invalid Date")

app = tk.Tk()
app.title("Age Calculator")

# Increase window size by 40%
app.geometry("420x420")

label_birthdate = tk.Label(app, text="Select your birthdate:")
label_birthdate.pack(pady=10)

# Date format changed to dd-mm-yyyy
birth_date_entry = DateEntry(app, date_pattern='dd-mm-yyyy')
birth_date_entry.pack(pady=10)

label_enddate = tk.Label(app, text="Select the end date for calculation (default is today):")
label_enddate.pack(pady=10)

# Date format changed to dd-mm-yyyy
end_date_entry = DateEntry(app, date_pattern='dd-mm-yyyy')
end_date_entry.pack(pady=10)

button_calculate = tk.Button(app, text="Calculate Age", command=calculate_age)
button_calculate.pack(pady=10)

label_result = tk.Label(app, text="")
label_result.pack(pady=20)

app.mainloop()
