import streamlit as st
import tensorflow as tf
from keras import models
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



df_minmax = pd.read_csv('minmax.csv')
# def model_predict(client):
#     predictions = model.predict(client)

#Функція для перевірки даних, завантажених з файлу, на наявність всіх необхідних колонок та відповідність типу двних 
def data_validation(df, rules):
    check = True
    for column, types in rules.items():
        if column not in df.columns:
            st.error(f"The {column} column is not found in the DataFrame.")
            check = False
            continue
        
        if not df[column].dropna().map(type).eq(types).all():
            st.error(f'Column {column} contains invalid data types.')
            check = False
            continue
        
        if not df[column].ge(0).all():
            st.error(f'Values in column {column} are less than 0.')
            check = False
        
    if not (df['is_tv_subscriber'].between(0, 1).all() and df['is_movie_package_subscriber'].between(0, 1).all()) :
        st.error(f'Values in column {column} are not in [0, 1].')
        check = False
    return check

# Функція для нормалізації
def df_normal(df_original):
    df_concat = pd.concat([df_minmax, df_original], ignore_index=True)

    scaler = MinMaxScaler()
    df_concat[numeric_cols] = scaler.fit_transform(df_concat[numeric_cols])

    df = df_concat.drop([0, 1])
    return df


#Функція для створення датафрейму з введених даних
def add_data():
    data = {
        'id': st.session_state.id,
        'is_tv_subscriber': st.session_state.is_tv_subscriber,
        'is_movie_package_subscriber': st.session_state.is_movie_package_subscriber,
        'subscription_age': st.session_state.subscription_age,
        'bill_avg': st.session_state.bill_avg,
        'reamining_contract': st.session_state.reamining_contract,
        'service_failure_count': st.session_state.service_failure_count,
        'download_avg': st.session_state.download_avg,
        'upload_avg': st.session_state.upload_avg,
        'download_over_limit': st.session_state.download_over_limit
    }
    df = pd.DataFrame([data])
    return df


# df_rules = {
#     'id': {'type': int},
#     'is_tv_subscriber': {'type': int, 'nan': False},
#     'is_movie_package_subscriber': {'type': int, 'nan': False},
#     'subscription_age': {'type': float, 'nan': False},
#     'bill_avg': {'type': int, 'nan': False},
#     'reamining_contract': {'type': float, 'nan': True},
#     'service_failure_count': {'type': int, 'nan': False},
#     'download_avg': {'type': float, 'nan': True},
#     'upload_avg':{'type': float, 'nan': True},
#     'download_over_limit': {'type': int, 'nan': False}
# }
numeric_cols = ['subscription_age', 'bill_avg', 'reamining_contract', 'service_failure_count','download_avg', 'upload_avg'] #'download_over_limit'

#Зразок типів даних в колонках
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

# Вибір між файлом та введенням даних
input_type = st.radio('Inputting customer data:', ['Upload the file', 'Enter from the keyboard'], index=0)

if input_type == 'Upload the file':
    #Завантаження файлу
    uploaded_file = st.file_uploader("Select the file...", type=["csv"])

    if uploaded_file is not None:
        # Відображення назви завантаженого файлу

        st.write("File name:", uploaded_file.name)

        # Читання файлу
        df_file = pd.read_csv(uploaded_file)
        if df_file.empty:
            st.warning("The uploaded CSV file is empty.")
        else:
            st.write(df_file)
            df = df_file

            # заміна порожніх значень 'reamining_contract' на 0
            if 'reamining_contract' in df.columns:
                df['reamining_contract'] = df['reamining_contract'].fillna(0)

            #Перевірка коректності даних. 
            if data_validation(df, df_type):

                #Попередня обробка даних
                df.drop('id', axis=1, inplace=True)

                df = df_normal(df)
            

                #Прогнозування моделі та вивід результатів
                #model_predict(df)

elif input_type == 'Enter from the keyboard':
    
    
    id = st.number_input('id:', step=1, min_value=1, key='id')

    col11, col12, col13 = st.columns(3)

    # створення трьох колонок
    with col11:
       is_tv_subscriber = st.radio('is_tv_subscriber', [0, 1], key='is_tv_subscriber')

    with col12:
       is_movie_package_subscriber = st.radio('is_movie_package_subscriber', [0, 1], key='is_movie_package_subscriber')

    with col13:
       subscription_age =  st.number_input('subscription_age:', min_value=0.00, value=0.00, key='subscription_age')

    col21, col22, col23 = st.columns(3)

    with col21:
       bill_avg = st.number_input('bill_avg:', step=1, min_value=0, key='bill_avg')
        
    with col22:
       reamining_contract = st.number_input('reamining_contract:', min_value=0.00, value=0.00, key='reamining_contract')

    with col23:
       service_failure_count = st.number_input('service_failure_count:', step=1, min_value=0, key='service_failure_count')
    
    col31, col32, col33 = st.columns(3)
       
    with col31:
       download_avg = st.number_input('download_avg:', min_value=0.00, value=0.00, key='download_avg')

    with col32:
       upload_avg = st.number_input('upload_avg:', min_value=0.00, value=0.00, key='upload_avg')

    with col33:
       download_over_limit = st.number_input('download_over_limit:', step=1, min_value=0, key='download_over_limit')
    

    # Кнопка для додавання даних
    if st.button("add a client"):
        if 'dataframe' not in st.session_state:
            st.session_state['dataframe'] = pd.DataFrame(columns=[
                'id', 'is_tv_subscriber', 'is_movie_package_subscriber',
                'subscription_age', 'bill_avg', 'reamining_contract',
                'service_failure_count', 'download_avg', 'upload_avg',
                'download_over_limit'])
            
        # Вивід датафрейму
        if id in st.session_state['dataframe']['id'].values: 
            st.error(f"id {id} already exists.")    
        else:
            new_data = add_data()
            st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], new_data], ignore_index=True)

    #Можливість видалення клієнта по id
    del_id = st.number_input('deletable identifier:', step=1, min_value=0)
    if st.button("delete the client"):
        if id in st.session_state['dataframe']['id'].values:
            st.session_state['dataframe'] = st.session_state['dataframe'].loc[st.session_state['dataframe']['id'] != del_id]    
    
    #Вивід даних
    if 'dataframe' in st.session_state:
        st.write("Current data:")
        st.write(st.session_state['dataframe'])

    


    if st.button("predict"):
        try:
            df_input = st.session_state['dataframe']
            df = df_input
            #Попередня обробка даних
            df.drop('id', axis=1, inplace=True)
            df = df_normal(df)
            #st.write(df)

            #Обробка моделлю

            # predictions = models_dic[type_model].predict(processed_image)
            # predicted_class = np.argmax(predictions)

            # # Вивід результатів
            # st.write("### Результати класифікації:")
            # st.write(f'Predicted class name: {class_names[predicted_class]}')
            # for label, prob in zip(class_names, predictions[0]):
            #     st.write(f"{label}: {prob * 100:.2f}%")
        except:
            st.warning("Please add the data.")



    








