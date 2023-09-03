import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import io
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib
import os


# Title and description
st.title("Automatic EDA Web App")
st.sidebar.subheader("Select Dataset")
# Sidebar options
data_option = st.sidebar.radio("Select Data Source", ("Seaborn Sample Dataset", "Upload Custom Data"))

# Data Importing
if data_option == "Upload Custom Data":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.sidebar.warning("This field is required.")
else:
    st.sidebar.subheader("Select Seaborn Dataset")
    dataset_names = sns.get_dataset_names()
    selected_dataset = st.sidebar.selectbox("Select a dataset", dataset_names)
    df = sns.load_dataset(selected_dataset)
nameofmodel = st.sidebar.text_input("Enter the name of the model")
if not nameofmodel:
    st.sidebar.warning("This field is required.")
# Display Data Description
st.subheader("Data Description")
if st.checkbox("Show Data"):
    st.write(df.head())
info_str = io.StringIO()
df.info(buf=info_str)
info_str.seek(0)
info_content = info_str.read()
if st.checkbox("Show Data Info"):
    st.write("DataFrame Information:")
    st.text(info_content)
if st.checkbox("Show Data Description"):
    st.write(df.describe())
































# Data Cleaning
st.subheader("Data Cleaning")
# User-defined functions for data cleaning
def handle_missing_values(df, column_name, method):
    if method == "None":
        return
    if method == "Drop":
        df.dropna(subset=[column_name], inplace=True)
    elif method == "Impute with Mean":
        if pd.api.types.is_numeric_dtype(df[column_name]):
            df[column_name].fillna(df[column_name].mean(), inplace=True)
    elif method == "Impute with Median":
        if pd.api.types.is_numeric_dtype(df[column_name]):
            df[column_name].fillna(df[column_name].median(), inplace=True)
    elif method == "Impute with Mode":
        if pd.api.types.is_categorical_dtype(df[column_name]) or df[column_name].dtype == 'object':
            mode_value = df[column_name].mode()[0]
            df[column_name].fillna(mode_value, inplace=True)
# User-defined function for data cleaning
def handle_missing_values(df, column_name, method):
    if method == "None":
        return
    if method == "Drop":
        df.dropna(subset=[column_name], inplace=True)
    elif method == "Impute with Mean":
        if pd.api.types.is_numeric_dtype(df[column_name]):
            df[column_name].fillna(df[column_name].mean(), inplace=True)
    elif method == "Impute with Median":
        if pd.api.types.is_numeric_dtype(df[column_name]):
            df[column_name].fillna(df[column_name].median(), inplace=True)
    elif method == "Impute with Mode":
        if pd.api.types.is_categorical_dtype(df[column_name]) or df[column_name].dtype == 'object':
            mode_value = df[column_name].mode()[0]
            df[column_name].fillna(mode_value, inplace=True)
    # Add more methods as needed

# Display the number and percentage of missing values in each column
st.subheader("Missing Values Analysis")
missing_values_count = df.isnull().sum()
missing_values_percentage = (missing_values_count / len(df)) * 100
missing_data_info = pd.DataFrame({
    "Missing Values Count": missing_values_count,
    "Percentage of Missing Values": missing_values_percentage
})
st.write(missing_data_info)

# Filter columns with missing values
columns_with_missing_values = missing_data_info[missing_data_info["Missing Values Count"] > 0].index.tolist()



for column in columns_with_missing_values:
    st.write(f"### {column}")
    # Determine the method list based on the column type
    if pd.api.types.is_numeric_dtype(df[column]):
        method_list = ["None", "Drop", "Impute with Mean", "Impute with Median", "Impute with Mode"]
    elif pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
        method_list = ["None", "Drop", "Impute with Mode"]
    
    clean_method = st.selectbox(f"Select Cleaning Method for {column}", method_list)
    if clean_method != "None":
        handle_missing_values(df, column, clean_method)

st.write("### Cleaned Data")
st.write(df.head())
missing_values_coun = df.isnull().sum()
missing_values_percentag = (missing_values_coun / len(df)) * 100
missing_data_inf = pd.DataFrame({
    "Missing Values Count": missing_values_coun,
    "Percentage of Missing Values": missing_values_percentag
})
st.write(missing_data_inf)




















# Function to generate visualizations based on selected graph names and column names
@st.cache_resource
def generate_visualizations(selected_graphs, column_names, df, selected_plots):
    graph_number = 1
    for plot_name in selected_plots:
        st.write(f"### Graph {graph_number}: {plot_name}")
        fig, ax = plt.subplots()
        if "Scatter Plot" in plot_name:
            x_col, y_col = plot_name.split("Scatter Plot between ")[1].split(" and ")
            ax.scatter(df[x_col], df[y_col])
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        elif "Line Chart" in plot_name:
            x_col, y_col = plot_name.split("Line Chart between ")[1].split(" and ")
            sns.lineplot(data=df, x=x_col, y=y_col)
        elif "Bar Chart" in plot_name:
            x_col, y_col = plot_name.split("Bar Chart for ")[1].split(" and ")
            sns.barplot(data=df, x=x_col, y=y_col)
        elif "Frequency Plot" in plot_name:
            column_name = plot_name.split(" Frequency Plot")[0]
            sns.countplot(data=df, x=column_name)
            ax.set_xlabel(column_name)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{column_name} Frequency Plot")
        elif "Distribution Plot" in plot_name:
            column_name = plot_name.split(" Distribution Plot")[0]
            sns.histplot(df[column_name], kde=True)
            ax.set_xlabel(column_name)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{column_name} Distribution Plot")
        st.pyplot(fig)
        graph_number += 1

# User input for plot types and column names
st.subheader("Data Visualization")
available_graphs = ["Scatter Plot", "Line Chart", "Bar Chart", "Frequency Plot", "Distribution Plot"]
selected_graphs = st.multiselect("Select Graphs to Generate", available_graphs)
selected_columns = st.multiselect("Select Columns for Visualization", df.columns)

possible_graphs = []
for graph_name in selected_graphs:
    for i, x_col in enumerate(selected_columns):
        for j, y_col in enumerate(selected_columns):
            if i < j:  # To avoid generating duplicate graphs by swapping x and y
                if graph_name == "Scatter Plot":
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        plot_name = f"Scatter Plot between {x_col} and {y_col}"
                        possible_graphs.append(plot_name)
                elif graph_name == "Line Chart":
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        plot_name = f"Line Chart between {x_col} and {y_col}"
                        possible_graphs.append(plot_name)
                elif graph_name == "Bar Chart":
                    if (pd.api.types.is_categorical_dtype(df[x_col]) or df[x_col].dtype == 'object') and pd.api.types.is_numeric_dtype(df[y_col]):
                        plot_name = f"Bar Chart for {x_col} and {y_col}"
                        possible_graphs.append(plot_name)
                elif graph_name == "Frequency Plot":
                    if pd.api.types.is_categorical_dtype(df[x_col]) or df[x_col].dtype == 'object':
                        plot_name = f"{x_col} Frequency Plot"
                        possible_graphs.append(plot_name)
                elif graph_name == "Distribution Plot":
                    plot_name = f"{x_col} Distribution Plot"
                    possible_graphs.append(plot_name)

selected_plots = st.multiselect("Select Plots to Generate", possible_graphs)

generate_visualizations(selected_graphs, selected_columns, df, selected_plots)










































# Data Encoding
st.subheader("Data Encoding")

# User-defined function for data encoding
def encode_categorical_columns(df, columns_to_encode, encoding_method):
    if encoding_method == "Label Encoding":
        label_encoder = LabelEncoder()
        for column in columns_to_encode:
            if pd.api.types.is_categorical_dtype(df[column]) or df[column].dtype == 'object':
                df[column] = label_encoder.fit_transform(df[column])
    elif encoding_method == "One-Hot Encoding":
        df = pd.get_dummies(df, columns=columns_to_encode, drop_first=True)
    # Add more encoding methods as needed

# Allow the user to select columns to encode
st.subheader("Select Columns for Data Encoding")
categorical_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
columns_to_encode = st.multiselect("Select Categorical Columns to Encode", categorical_columns)

# Allow the user to choose an encoding method
encoding_method = st.radio("Select Encoding Method", ("Label Encoding", "One-Hot Encoding"))

# Call the encoding function immediately when the user makes their selections
if encoding_method and columns_to_encode:
    encode_categorical_columns(df, columns_to_encode, encoding_method)

# Display the encoded data
st.write("### Encoded Data")
st.write(df.head())
























# Correlation Heatmap
st.subheader("Correlation Heatmap")
encode_categorical_columns(df, categorical_columns, "Label Encoding")
if st.checkbox("Show Correlation Heatmap"):
    corr_matrix = df.corr()
    st.write(corr_matrix)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)
    st.pyplot(fig)












# Generate a summary of the first few rows of your DataFrame
df_head_summary = df.head().to_string()

# Add this summary to your PDF report data

# Generate a summary of df.describe()
df_describe_summary = df.describe().to_string()

# Generate a summary of df.info()
df_info_summary = df.info(verbose=False, buf=None, max_cols=None)

# Add these summaries to your PDF report data










# Machine Learning
st.subheader("Machine Learning")
asdf = {}
encode_categorical_columns(df, categorical_columns, "Label Encoding")
# User-defined functions for machine learning
def train_classification_model(df, target_column, algorithm):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    test_size = st.slider("Select Test Size (as a fraction)", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif algorithm == "Decision Tree Classifier":
        model = DecisionTreeClassifier()
    elif algorithm == "Random Forest Classifier":
        model = RandomForestClassifier()
    # Add more classification models as needed

    model.fit(X_train, y_train)
    return model, X_test, y_test

def train_regression_model(df, target_column, algorithm):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    test_size = st.slider("Select Test Size (as a fraction)", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if algorithm == "Linear Regression":
        model = LinearRegression()
    elif algorithm == "Decision Tree Regressor":
        model = DecisionTreeRegressor()
    elif algorithm == "Random Forest Regressor":
        model = RandomForestRegressor()
    # Add more regression models as needed

    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model):
    joblib.dump(model, nameofmodel+'.pkl')

def generate_report(model, X_test, y_test, algorithm):
    y_pred = model.predict(X_test)
    report = []

    if algorithm in ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        asdf["MAE"] = mae
        asdf["MSE"] = mse
        asdf["RMSE"] = rmse
        asdf["R2"] = r2

        report.append(f"Mean Absolute Error (MAE): {mae:.4f}")
        report.append(f"Mean Squared Error (MSE): {mse:.4f}")
        report.append(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        report.append(f"R-squared (R2): {r2:.4f}")
    elif algorithm in ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"]:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multi-class
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)

        report.append(f"Accuracy: {accuracy:.4f}")
        report.append(f"Precision (Macro): {precision:.4f}")
        report.append(f"Recall (Macro): {recall:.4f}")
        cm_formatted = "\n".join(" ".join(str(cell) for cell in row) for row in cm)
        report.append(f"Confusion Matrix:\n{cm_formatted}")
        asdf["Accuracy"] = accuracy
        asdf["Precision"] = precision
        asdf["Recall"] = recall
        asdf["F1"] = f1
        asdf["Confusion Matrix"] = cm_formatted
    return report



algorithm = st.selectbox("Select Algorithm", ("Classification", "Regression"))
if algorithm == "Classification":
    target_column = st.selectbox("Select Target Column for Classification", categorical_columns)
    classification_models = ["Logistic Regression", "Decision Tree Classifier", "Random Forest Classifier"]
    classification_model = st.selectbox("Select Classification Model", classification_models)        
    st.write("Training Classification Model...")
    clf_model, X_test, y_test = train_classification_model(df, target_column, classification_model)
    save_model(clf_model)
    aslam = clf_model
    st.write("Classification Model Trained Successfully and Saved as trained_model.pkl")
    st.write("Generating Classification Report...")
    eport = generate_report(clf_model, X_test, y_test, classification_model)    

elif algorithm == "Regression":
    target_column = st.selectbox("Select Target Column for Regression", [col for col in df.columns if col not in categorical_columns])
    regression_models = ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
    regression_model = st.selectbox("Select Regression Model", regression_models)        
    st.write("Training Regression Model...")
    reg_model, X_test, y_test = train_regression_model(df, target_column, regression_model)
    save_model(reg_model)
    aslam = reg_model
    st.write("Regression Model Trained Successfully and Saved as trained_model.pkl")
    st.write("Generating Regression Report...")
    eport = generate_report(reg_model, X_test, y_test, regression_model)    

# Add problem type and algorithm information to your PDF report data





def generate_and_save_html_report(df, metrics_info, save_path):
    report_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background: blanchedalmond;
                text-align: center;
            }
            h1 {
                color: #333;
            }
            p {
                margin-top: 10px;
                margin-bottom: 10px;
            }
            table {
                border-collapse: collapse;
                margin: auto;
                min-width: 70%;
            }
            th, td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            th {
                font-weight: 700;
            }
        </style>
    </head>
    <body>
        <h1>Data Analysis Report</h1>
        <!-- Add DataFrame Information -->
        <h2>Dataset Information</h2>
    """

    # Add df.head() information
    report_html += metrics_info["head"].to_html(classes="table table-striped table-bordered")

    report_html += """
        <h2>Dataset Info</h2>
    """

    # Read the content from the buffer


    # Append the content to the HTML report
    report_html += f"""
        <pre>{info_content}</pre>
    """
    report_html += """
        <!-- Add Metrics Information -->
        <h2>Dataset Description</h2>
    """

    # Add df.describe() information
    report_html += metrics_info["describe"].to_html(classes="table table-striped table-bordered")

    report_html += """
        <!-- Add Metrics Information -->
        <h2>Metrics</h2>
    """
    if algorithm == "Classification":
        confusion_matrix_html = "<table border='1'>"
        confusion_matrix_data = metrics_info["Confusion Matrix"].split("\n")
        for row in confusion_matrix_data:
            confusion_matrix_html += "<tr>"
            values = row.split()
            for value in values:
                confusion_matrix_html += f"<td>{value}</td>"
            confusion_matrix_html += "</tr>"
        confusion_matrix_html += "</table>"
        asdfgh = """
                    <tr>
                        <td>Confusion Matrix</td>
                        <td>{}</td>
                    </tr>
                """.format(confusion_matrix_html)
        report_html += f"""
        <p>Problem Type: {algorithm}</p>
        <p>Algorithm Used: {classification_model if algorithm=="Classification" else regression_model}</p>
        <table>
            <tr>
                <th>Metric Name</th>
                <th>Metric Value</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{metrics_info["Accuracy"]}</td>
            </tr>
            <tr>
                <td>Precision (Macro)</td>
                <td>{metrics_info["Precision"]}</td>
            </tr>
            <tr>
                <td>Recall (Macro)</td>
                <td>{metrics_info["Recall"]}</td>
            </tr>
            <tr>
                <td>F1 Score (Macro)</td>
                <td>{metrics_info["F1"]}</td>
            </tr>
            <tr>
                {asdfgh}
            </tr>
            </table>
            """
    else:
        report_html += f"""
        <p>Problem Type: {algorithm}</p>
        <p>Algorithm Used: {classification_model if algorithm=="Classification" else regression_model}</p>
        <table>

            <tr>
                <th>Metric Name</th>
                <th>Metric Value</th>
            </tr>   
            <tr>
                <td>Mean Absolute Error (MAE)</td>
                <td>{metrics_info["MAE"]}</td>
            </tr>
            <tr>
                <td>Mean Squared Error (MSE)</td>
                <td>{metrics_info["MSE"]}</td>
            </tr>
            <tr>
                <td>Root Mean Squared Error (RMSE)</td>
                <td>{metrics_info["RMSE"]}</td>
            </tr>
            <tr>
                <td>R-squared (R2)</td>
                <td>{metrics_info["R2"]}</td>
            </tr>
        </table>
        """
    report_html += """
    </body>
    </html>
    """

    # Save the HTML report to the specified folder
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report_html)

    return report_html


# Get DataFrame information
df_nfo = {
    "head": df.head(),
    "describe": df.describe(),
}
df_info = {**df_nfo, **asdf}
# Define the folder path to save the HTML report
report_folder_path = ""


# Generate and save the HTML report
report_filename = f"{nameofmodel}.html"
report_full_path = os.path.join(report_folder_path, report_filename)

# Create a button to download the .pkl file
file_path = ""  # Replace with the actual path to your file
file_name = nameofmodel+'.pkl' # Replace with the name of your file

# Create a Streamlit button
st.subheader("Download Trained Model")
if st.button("Download Trained model"):
    with open(os.path.join(file_path, file_name), "rb") as file:
        file_bytes = file.read()
    st.download_button(label="Click to Download", data=file_bytes, file_name=f"{nameofmodel}.pkl", key="download_file")
report_html = generate_and_save_html_report(eport, df_info, report_full_path)
st.subheader("HTML Report")
with open(report_full_path, "r", encoding="utf-8") as file:
    html_content = file.read()
st.components.v1.html(html_content, width=800, height=1500)

# Provide a download link for the user to download the saved HTML report
st.subheader("Download HTML Report")

if os.path.exists(report_full_path):
    with open(report_full_path, "rb") as f:
        report_data = f.read()
    st.download_button("Download HTML Report", report_data, file_name=report_filename, key="html_report")
else:
    st.write("No HTML report found. Generate the report first.")





