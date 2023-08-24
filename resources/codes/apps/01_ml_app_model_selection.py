import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import plotly.express as px

st.title('My ML App')

uploaded_file = st.sidebar.file_uploader('Upload a CSV file', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    categorical_cols = df.select_dtypes(include='object').columns
    st.write('Categorical columns:', categorical_cols)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns 
    st.write('Numerical columns:', numeric_cols)
    
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])
        
    X_cols = st.sidebar.multiselect('Select feature columns', df.columns)
    y_col = st.sidebar.selectbox('Select target column', df.columns)
    
    test_size = st.sidebar.slider('Test size (%)', min_value=10, max_value=40, value=20)
    random_state = 42
    
    problem_type = st.sidebar.selectbox('Regression or Classification', ['Regression', 'Classification'])
    
    if problem_type == 'Regression':
        models = ['Linear Regression', 'Ridge', 'Lasso', 'Polynomial', 'SVR', 'Decision Tree', 'KNN', 'Random Forest', 'Gradient Boosting', 'XGBoost']
        
        model_metrics = {
            'Linear Regression': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'Ridge': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},  
            'Lasso': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'Polynomial': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'SVR': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'Decision Tree': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'KNN': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'Random Forest': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'Gradient Boosting': {'MSE': mean_squared_error, 'MAE': mean_absolute_error},
            'XGBoost': {'MSE': mean_squared_error, 'MAE': mean_absolute_error}
        }
        
    elif problem_type == 'Classification':
        models = ['Logistic Regression', 'SVC', 'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'Gradient Boosting', 'XGBoost']
        
        model_metrics = {
            'Logistic Regression': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score},
            'SVC': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score},
            'KNN': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score},
            'Decision Tree': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score},
            'Random Forest': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score},
            'Naive Bayes': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score},
            'Gradient Boosting': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score}, 
            'XGBoost': {'Accuracy': accuracy_score, 'Precision': precision_score, 'Recall': recall_score, 'F1': f1_score}
        }
        
    X = df[X_cols]
    y = df[y_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state) 
    
    model_selection = st.sidebar.selectbox('Select a model', models)
    
    if model_selection == 'Linear Regression':
        model = LinearRegression()
        
    elif model_selection == 'Ridge':
        model = Ridge()
        
    elif model_selection == 'Lasso':
        model = Lasso()
        
    elif model_selection == 'Polynomial':
        model = PolynomialRegression()
        
    elif model_selection == 'SVR':
        model = SVR()
        
    elif model_selection == 'Decision Tree':
        model = DecisionTreeRegressor()
        
    elif model_selection == 'KNN':
        model = KNeighborsRegressor()
        
    elif model_selection == 'Random Forest':
        model = RandomForestRegressor()
        
    elif model_selection == 'Gradient Boosting':
        model = GradientBoostingRegressor()
        
    elif model_selection == 'XGBoost':
        model = XGBRegressor()
        
    elif model_selection == 'Logistic Regression':
        model = LogisticRegression()
        
    elif model_selection == 'SVC':
        model = SVC()
        
    elif model_selection == 'KNN':
        model = KNeighborsClassifier()
        
    elif model_selection == 'Decision Tree':
        model = DecisionTreeClassifier()
        
    elif model_selection == 'Random Forest':
        model = RandomForestClassifier()
        
    elif model_selection == 'Naive Bayes':
        model = GaussianNB()
        
    elif model_selection == 'Gradient Boosting':
        model = GradientBoostingClassifier()
        
    elif model_selection == 'XGBoost':
        model = XGBClassifier()
        
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metric_names = model_metrics[model_selection].keys()
    metric_values = [metric(y_test, y_pred) for metric in model_metrics[model_selection].values()]
    
    metrics_df = pd.DataFrame({'Metric': metric_names, 'Value': metric_values})
    
    st.write(metrics_df)
    
    fig = px.line(metrics_df, x='Metric', y='Value')
    st.plotly_chart(fig)
    
    st.write('Best model:', model_selection)