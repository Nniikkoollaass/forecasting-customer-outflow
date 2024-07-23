import streamlit as st
import tensorflow as tf
from keras import models
import pandas as pd
import numpy as np





# def model_predict(client):
#     predictions = model.predict(client)
def add_data():
    data = {
        'Name': st.session_state['name'],
        'Age': st.session_state['age'],
        'Salary': st.session_state['salary'],
        'Is Active': st.session_state['is_active']
    }
    df = pd.DataFrame([data])
    return df
def data_validation(df, rules):
    for column, types in rules.items():
        if column not in df.columns:
            st.error(f"The {column} column is not found in the DataFrame.")
            continue
        
        if not df[column].map(type).eq(types).all():
            st.error(f'Column {column} contains invalid data types.')
            continue
        
        if not df[column].ge(0).all():
            st.error(f'Values in column {column} are less than 0.')
        
    if not (df['is_tv_subscriber'].between(0, 1).all() and df['is_movie_package_subscriber'].between(0, 1).all()) :
        st.error(f'Values in column {column} are not in [0, 1].')

# df_rules = {
#     'id': {'type': int},
#     'is_tv_subscriber': {'type': int, 'range': (0, 1)},
#     'is_movie_package_subscriber': {'type': int, 'range': (0, 1)},
#     'subscription_age': {'type': float, 'range': (0,)},
#     'bill_avg': {'type': int, 'range': (0,)},
#     'reamining_contract': {'type': float, 'range': (0,)},
#     'service_failure_count': {'type': int, 'range': (0,)},
#     'download_avg': {'type': float, 'range': (0)},
#     'upload_avg':{'type': float, 'range': (0,)},
#     'download_over_limit': {'type': int, 'range': (0,)}
# }
df_type = {
    'id': int,
    'is_tv_subscriber': int,
    'is_movie_package_subscriber': int,
    'subscription_age': float,
    'bill_avg': int,
    'reamining_contract': float,
    'service_failure_count': int,
    'download_avg': float,
    'upload_avg': float,
    'download_over_limit': int
}
 
#Завантаження моделі
#model = models.load_model('model.h5')

st.title('Predicting customer churn for a telecoms company')



input_type = st.sidebar.radio('Inputting customer data:', ['Upload the file', 'Enter from the keyboard'], index=0)

if input_type == 'Upload the file':
    uploaded_file = st.file_uploader("Select the file...", type=["csv"])

    if uploaded_file is not None:
        # Відображення назви завантаженого файлу

        st.write("File name:", uploaded_file.name)

        # Читання файлу
        df = pd.read_csv(uploaded_file)
        if df.empty:
            st.warning("The uploaded CSV file is empty.")
        

        #Перевірка коректності даних
        data_validation(df, df_type)

        #Попередня обробка даних

        #Прогнозування моделі та вивід результатів
        #model_predict(df)

elif input_type == 'Enter from the keyboard':
    # Создание трех колонок
    
    col11, col12, col13 = st.columns(3)
    col21, col22, col23 = st.columns(3)
    col31, col32, col33 = st.columns(3)
    # num_rows = 3
    # num_cols = 3
    # cols = []
    # for i in range(num_rows):
    #     cols[i] = st.columns(num_cols) 
        
            

    # Размещение виджетов в колонках
    with col11:
       is_tv_subscriber = st.radio('is_tv_subscriber', [0, 1])

    with col12:
       is_movie_package_subscriber = st.radio('is_movie_package_subscriber', [0, 1])

    with col13:
       subscription_age =  st.number_input('subscription_age:', min_value=0.00, value=0.00)

    with col21:
       bill_avg = st.number_input('bill_avg:', step=1, min_value=0)
        
    with col22:
       reamining_contract = st.number_input('reamining_contract:', min_value=0.00, value=0.00)

    with col23:
       service_failure_count = st.number_input('service_failure_count:', step=1, min_value=0)
    
    with col31:
       download_avg = st.number_input('download_avg:', min_value=0.00, value=0.00)

    with col32:
       upload_avg = st.number_input('upload_avg:', min_value=0.00, value=0.00)

    with col33:
       download_over_limit = st.number_input('download_over_limit:', step=1, min_value=0)
    

    # Кнопка для добавления данных
    if st.button("Add data"):
        if 'dataframe' not in st.session_state:
            st.session_state['dataframe'] = pd.DataFrame(columns=[
                'id', 'is_tv_subscriber', 'is_movie_package_subscriber',
                'subscription_age', 'bill_avg', 'reamining_contract',
                'service_failure_count', 'download_avg', 'upload_avg',
                'download_over_limit'])
            
        new_data = add_data()
        st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], new_data], ignore_index=True)

    # Показать DataFrame
    if 'dataframe' in st.session_state:
        st.write("Текущие данные:")
        st.write(st.session_state['dataframe'])










