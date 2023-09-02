import streamlit as st
import pandas as pd
import pandas_profiling
# from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
from skimpy import skim
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. App Name
st.title('Machine Learning Tool for Beginners')

# 2. Upload Data or Choose Sample Data
data_source = st.sidebar.radio('Choose your data source', ['Upload my own data', 'Use a sample dataset'])

if data_source == 'Upload my own data':
    uploaded_file = st.sidebar.file_uploader("Upload your data", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

elif data_source == 'Use a sample dataset':
    dataset_name = st.sidebar.selectbox('Choose a sample dataset', ['tips', 'titanic'])
    if dataset_name == 'tips':
        df = sns.load_dataset('tips')
    else:
        df = sns.load_dataset('titanic')

# Only run the following if df exists (either from uploading or from sample dataset)
if 'df' in locals():

    # 3. Basic EDA using Pandas Profiling
    if st.sidebar.radio('Would you like to do EDA?', ['Yes', 'No']) == 'Yes':
        st.write('EDA Overview')
        pr = df.profile_report()
        st.write(pr)
        

    # 5. Separating columns based on dtype
    st.write('Dataframe Overview')
    st.write(df.head())
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['float64', 'int64']).columns.tolist()
    st.write(f'Numeric Columns: {numeric_cols}')
    st.write(f'Categorical Columns: {categorical_cols}')

    # 6. Selecting predictor columns
    selected_cols = st.multiselect('Select columns for predictor X', df.columns)

    # 7 & 8. Encoding categorical variables
    encoding_method = st.sidebar.selectbox('Select an encoding method', ['Label Encoding', 'One Hot Encoding'])
    if encoding_method == 'Label Encoding':
        le = LabelEncoder()
        for col in selected_cols:
            if col in categorical_cols:
                df[col] = le.fit_transform(df[col])
    elif encoding_method == 'One Hot Encoding':
        df = pd.get_dummies(df, columns=categorical_cols)

    # 9. Selecting label column
    label_col = st.selectbox('Select one column as label (y)', df.columns)

    # 10 & 11. Classifying problem type
    if label_col in categorical_cols:
        st.sidebar.write("This is a classification problem")
        st.markdown("<h1 style='color: green;'>This is a Classification Problem</h1>", unsafe_allow_html=True)
    else:
        st.sidebar.write("This is a regression problem")
        st.markdown("<h1 style='color: red;'>This is a Regression Problem</h1>", unsafe_allow_html=True)
