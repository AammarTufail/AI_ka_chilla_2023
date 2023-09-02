import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def main():
    st.title("Streamlit EDA App")

    # Upload dataset or select sample data
    data_source = st.sidebar.radio("Select Data Source", ["Upload Dataset", "Use Sample Data"])
    if data_source == "Upload Dataset":
        uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
    else:
        sample_dataset_name = st.sidebar.selectbox("Select a sample dataset", ["tips", "titanic"])
        data = sns.load_dataset(sample_dataset_name)

    if data is not None:

        # EDA Section
        if st.sidebar.checkbox("Do you want to perform basic EDA analysis?"):
            st.write("### Dataset Overview:")
            st.write(data.head())
            st.write("### Basic Statistics:")
            st.write(data.describe())
            if len(data.columns) < 5:
                st.write("### Pairplot:")
                fig = sns.pairplot(data)
                st.pyplot(fig)
            else:
                st.write("### Correlation Matrix:")
                numeric_data = data.select_dtypes(include=[np.number])
                corr_matrix = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, ax=ax, cmap="coolwarm")
                st.pyplot(fig)

        # Plotting Section
        if st.sidebar.checkbox("Do you want to plot the data?"):
            st.sidebar.subheader("Plotting Options")
            columns_to_plot = st.sidebar.multiselect("Select columns to plot", data.columns)
            # Plotting based on selected columns (you can further modify this as per your requirements)
            if len(columns_to_plot) == 1:
                if data[columns_to_plot[0]].dtype in [np.float64, np.int64]:
                    fig = px.histogram(data, x=columns_to_plot[0])
                    st.plotly_chart(fig)
                elif data[columns_to_plot[0]].dtype == object:
                    fig = px.bar(data, x=columns_to_plot[0])
                    st.plotly_chart(fig)
            elif len(columns_to_plot) == 2:
                col1_dtype = data[columns_to_plot[0]].dtype
                col2_dtype = data[columns_to_plot[1]].dtype

                if col1_dtype in [np.float64, np.int64] and col2_dtype in [np.float64, np.int64]:
                    fig = px.scatter(data, x=columns_to_plot[0], y=columns_to_plot[1])
                    st.plotly_chart(fig)
                elif col1_dtype == object and col2_dtype == object:
                    fig = px.bar(data, x=columns_to_plot[0], color=columns_to_plot[1])
                    st.plotly_chart(fig)
                elif col1_dtype == object and col2_dtype in [np.float64, np.int64]:
                    fig = px.bar(data, x=columns_to_plot[0], y=columns_to_plot[1])
                    st.plotly_chart(fig)

        # ML Task Selection
        st.sidebar.subheader("Machine Learning Tasks")
        X_columns = st.sidebar.multiselect("Select feature columns (X)", data.columns)
        y_column = st.sidebar.selectbox("Select target column (y)", data.columns)

        # Encoding Section
        st.sidebar.subheader("Data Encoding")
        st.write(data.dtypes)
        categorical_cols = data[X_columns].select_dtypes(include=['object', 'category']).columns.tolist()
        encoding_method = st.sidebar.selectbox("Select an encoding method", ["None", "Label Encoding", "One-Hot Encoding"])

        if encoding_method == "Label Encoding":
            label_encoder = LabelEncoder()
            for col in categorical_cols:
                data[col] = label_encoder.fit_transform(data[col])
            st.write("Data after Label Encoding:")
            st.write(data.head())
        elif encoding_method == "One-Hot Encoding":
            data = pd.get_dummies(data, columns=categorical_cols)
            st.write("Data after One-Hot Encoding:")
            st.write(data.head())

        # Train-test split ratio
        split_ratio = st.sidebar.slider("Select train-test split ratio (%)", 10, 90, 80)
        st.sidebar.text(f"Train set size: {split_ratio}%")
        st.sidebar.text(f"Test set size: {100-split_ratio}%")

        if y_column in data.columns:
                task_type = st.sidebar.radio("Select Task Type", ["Regression", "Classification"])
                X = data[X_columns]
                y = data[y_column]

                # Scaling
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Splitting data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(100 - split_ratio) / 100.0, random_state=42)

                if task_type == "Regression":
                    st.sidebar.warning("This is a Regression Problem!")

                    # Regression models selection
                    regression_models = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(),
                        "Lasso Regression": Lasso(),
                        "Support Vector Regression": SVR(),
                        "Random Forest Regressor": RandomForestRegressor(),
                        "Gradient Boosting Regressor": GradientBoostingRegressor()
                    }

                    selected_regression_models = st.sidebar.multiselect("Select Regression Models", list(regression_models.keys()))

                    if selected_regression_models:
                        regression_results = {}
                        for model_name in selected_regression_models:
                            model = regression_models[model_name]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            r2 = r2_score(y_test, y_pred)
                            regression_results[model_name] = [mse, r2]

                        best_regression_model_name = max(regression_results, key=lambda k: regression_results[k][1])
                        worst_regression_model_name = min(regression_results, key=lambda k: regression_results[k][1])

                        st.write(f"### Best Regression Model: {best_regression_model_name}")
                        st.write(f"Mean Squared Error: {regression_results[best_regression_model_name][0]}")
                        st.write(f"R2 Score: {regression_results[best_regression_model_name][1]}")

                        st.write(f"### Worst Regression Model: {worst_regression_model_name}")
                        st.write(f"Mean Squared Error: {regression_results[worst_regression_model_name][0]}")
                        st.write(f"R2 Score: {regression_results[worst_regression_model_name][1]}")

                elif task_type == "Classification":
                    st.sidebar.warning("This is a Classification Problem!")

                    # Encoding the target column if it's categorical
                    le = LabelEncoder()
                    y = le.fit_transform(y)

                    # Classification models selection
                    classification_models = {
                        "Logistic Regression": LogisticRegression(),
                        "Random Forest Classifier": RandomForestClassifier(),
                        "Gradient Boosting Classifier": GradientBoostingClassifier(),
                        "Support Vector Classifier": SVC(),
                        "K-Nearest Neighbors": KNeighborsClassifier(),
                        "Decision Tree Classifier": DecisionTreeClassifier()
                    }

                    selected_classification_models = st.sidebar.multiselect("Select Classification Models", list(classification_models.keys()))

                    if selected_classification_models:
                        classification_results = {}
                        for model_name in selected_classification_models:
                            model = classification_models[model_name]
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')  # assuming multiclass, change if needed
                            classification_results[model_name] = [acc, f1]

                        best_classification_model_name = max(classification_results, key=lambda k: classification_results[k][0])
                        worst_classification_model_name = min(classification_results, key=lambda k: classification_results[k][0])

                        st.write(f"### Best Classification Model: {best_classification_model_name}")
                        st.write(f"Accuracy: {classification_results[best_classification_model_name][0]}")
                        st.write(f"F1-Score: {classification_results[best_classification_model_name][1]}")

                        st.write(f"### Worst Classification Model: {worst_classification_model_name}")
                        st.write(f"Accuracy: {classification_results[worst_classification_model_name][0]}")
                        st.write(f"F1-Score: {classification_results[worst_classification_model_name][1]}")

if __name__ == "__main__":
    main()
