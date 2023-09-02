import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import plotly.express as px

# Set the title
st.title('My Machine Learning Application!')


# Include sidebar with credentials
with st. sidebar:
            st.markdown('My ML App (V 0.1)')
            st.markdown(""" 
                        #### Let's connect:
                        [Kamran Feroz](https://www.linkedin.com/in/kamranferoz/)
            
                        #### Powered by:
                        [OpenAI](https://openai.com/)
                        [Langchain](https://github.com/hwchase17/langchain)\n
            
                        #### Source code:
                        [My ML App!](https://github.com/kamranferoz/myML)
                        """)
            st.markdown(
                "<style>#MainMenu{visibility:hidden;}</style>",
                unsafe_allow_html=True)



uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    cat_columns = data.select_dtypes(include=['object']).columns.tolist()
    num_columns = data.select_dtypes(exclude=['object']).columns.tolist()
    st.write(f'Categorical Columns: {cat_columns}')
    st.write(f'Numerical Columns: {num_columns}')

    # Encode categorical columns
    label_encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    st.write('Encoded Data:')
    st.write(data.head())

    # Feature and target selection
    selected_features = st.sidebar.multiselect('Select features for ML:', data.columns.tolist())
    y_column = st.sidebar.selectbox('Select target column:', data.columns.tolist())
    test_size = st.sidebar.slider('Test Train Split %:', 0.1, 0.9, 0.2)
    problem_type = st.sidebar.selectbox('Problem Type:', ['Regression', 'Classification'])


    if problem_type == 'Regression':
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'SVR': SVR(),
            'DT Regressor': DecisionTreeRegressor(),
            'KNN Regressor': KNeighborsRegressor(),
            'RF Regressor': RandomForestRegressor(),
            'GB Regressor': GradientBoostingRegressor(),
            'XGBoost Regressor': xgb.XGBRegressor()
        }

    else:
        models = {
            'Logistic': LogisticRegression(max_iter=1000),
            'SVC': SVC(),
            'KNN Classifier': KNeighborsClassifier(),
            'DT Classifier': DecisionTreeClassifier(),
            'RF Classifier': RandomForestClassifier(),
            'NB Classifier': GaussianNB(),
            'GB Classifier': GradientBoostingClassifier(),
            'XGBoost Classifier': xgb.XGBClassifier()
        }

    if st.button('Run Models'):
        X = data[selected_features]
        y = data[y_column]

        if "Polynomial" in models:
            poly = PolynomialFeatures(degree=2)
            X = poly.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        all_metrics = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if problem_type == 'Regression':
                metrics = {
                    'Mean Squared Error': mean_squared_error(y_test, y_pred),
                    'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
                    'R2 Score': r2_score(y_test, y_pred)
                }
            else:
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average="weighted"),
                    'Recall': recall_score(y_test, y_pred, average="weighted"),
                    'F1 Score': f1_score(y_test, y_pred, average="weighted"),
                    'Classification Report': classification_report(y_test, y_pred, output_dict=True)
                }

            all_metrics[model_name] = metrics
    
            
                
        
        best_model = max(all_metrics, key=lambda k: all_metrics[k].get('R2 Score', 0) or all_metrics[k].get('Accuracy', 0))
        st.write(f"Best Model: {best_model}")

        metrics_df = pd.DataFrame(all_metrics).T
        st.write(metrics_df)

        fig = px.bar(metrics_df, y=metrics_df.columns, title='Model Evaluation', barmode='group')
        st.plotly_chart(fig)
