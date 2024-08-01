import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score

#Фунція для обробки даних моделю та виводу результатів
def model_predict(df, origin_df, model):
    #прогрес бар
    progress_bar = st.progress(0)
    i = 0 
    total_operations = len(origin_df)
    percent_text = st.empty()
    #класи для виведення тексту
    #class_names = ['low', 'high']
    #перетворення df для коректної обробки моделлю xgboost
    dmat = xgb.DMatrix(df)
    pred = model.predict(dmat)
    
    #Створення та перейменування колонки з результатами
    churn = pd.DataFrame(pred)
    churn.columns = ['pred_churn']
    #Об'єднання origin_df з результатом
    origin_churn_df = pd.concat([origin_df, churn], axis=1)
    #Виведення кожного id з надписом The customer has a low/high probability of churn
    df_report = pd.DataFrame()
    for id, client in origin_churn_df.iterrows():
        i += 1
        row = pd.DataFrame({'id': [client['id']], 'pred_churn': [f"The customer has a {client['pred_churn']:.2%} probability of churn"]})

        df_report = pd.concat([df_report, row], ignore_index=True)
        #прогрес бар
        progress_percentage = int(i / total_operations * 100)
        progress_bar.progress(progress_percentage)
        percent_text.text(f'{progress_percentage}%')
    st.dataframe(df_report.to_dict('records'))
    origin_churn_df = pd.concat([origin_df, np.round(churn).astype(int)], axis=1)
    return origin_churn_df

#Функція для перевірки даних, завантажених з файлу, на наявність всіх необхідних колонок та відповідність типу двних 
def data_validation(df, rules):
    #Якщо check стане дорівнювати False програма не стане далі обробляти дані
    #rules - шаблон типів даних
    check = True
    for column, types in rules.items():
        #Перевірка чи э кожна колонка rules в датафреймі
        if column not in df.columns:
            st.warning(f"The {column} column is not found in the DataFrame.")
            check = False
            continue
        #Перевірка чи кожен елемент в клонці df[column] відповідає типу даних types
        if not df[column].map(type).eq(types).all():
            st.warning(f'Column {column} contains invalid data types.')
            check = False
            continue
        #Перевірка чи кожен елемент в клонці df[column] більший за 0
        if not df[column].ge(0).all():
            st.warning(f'Values in column {column} are less than 0.')
            check = False
        #Перевірка чи кожен елемент в клонках df['is_tv_subscriber'] та df['is_movie_package_subscriber'] між [0, 1]
    if check:  
        if not (df['is_tv_subscriber'].between(0, 1).all() and df['is_movie_package_subscriber'].between(0, 1).all()) :
            st.warning(f'Values in column {column} are not in [0, 1].')
            check = False
    #Повертає True або False
    return check

# Функція для нормалізації
def df_normal(df_original, df_minmax):
    #df_minmax - датафрейм з мінімумами та максимумами даних до нормалізації на яких навчалась модель
    #Колонки для нормалізації
    numeric_cols = ['is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age', 'bill_avg', 'download_avg', 'upload_avg']
    #Нормалізація
    scaler = MinMaxScaler()
    scaler.fit(df_minmax[numeric_cols])
    df_original[numeric_cols] = scaler.transform(df_original[numeric_cols])
    #Повертає нормалізований df_original
    return df_original

#Функція для створення датафрейму з введених даних
def add_data():
    # Створення словнику з введених з клавіатури даних збережених в st.session_state
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
    #Повертає датафрейм з однією строкою
    return df

def main():
    
    #Зразок типів даних в колонках
    df_type = {
        'id': int,
        'is_tv_subscriber': int,
        'is_movie_package_subscriber': int,
        'subscription_age': float,
        'bill_avg': int,
        #'reamining_contract': float,
        #'service_failure_count': int,
        'download_avg': float,
        'upload_avg': float,
        #'download_over_limit': int
    }
    
    #Завантаження моделі
    model = xgb.Booster()
    model.load_model('xgb_model.json')

    #дані для нормалізації
    df_minmax = pd.read_csv('minmax.csv')
    
    st.title('Predicting customer churn for a telecoms company')

    # Вибір між файлом та введенням даних
    #input_type - змінна зберігає зроблений вибір
    input_type = st.radio('Inputting customer data:', ['Upload the file', 'Enter from the keyboard'], index=0)

    #Якщо орано роботу з файлом
    if input_type == 'Upload the file':
        #Завантаження файлу
        uploaded_file = st.file_uploader("Select the file...", type=["csv"])

        #Якщо файл завантажений
        if uploaded_file is not None:
            # Відображення назви завантаженого файлу
            st.write("File name:", uploaded_file.name)
            # Читання файлу
            df_file = pd.read_csv(uploaded_file)
            #Якщо файл пустий, вивести:
            if df_file.empty:
                st.warning("The uploaded CSV file is empty.")
            #Якщо файл не пустиий почати роботу
            else:
                #Вивести на екран, початкові дані
                st.dataframe(df_file.to_dict('records'))
                #Якщо натиснуто кнопку predict, полчати обробку 
                if st.button("predict"):
                    df = df_file

                    if 'download_avg' in df.columns and 'upload_avg' in df.columns and 'subscription_age' in df.columns:
                        df.loc[df['subscription_age'] < 0, 'subscription_age'] = 0
                        df['download_avg'].fillna(0, inplace=True)
                        df['upload_avg'].fillna(0, inplace=True)
                    
                    #Перевірка коректності даних 
                    #Якщо функція повернула True, продовжити
                    if data_validation(df, df_type):
                        #Якщо в наданих даних є зайві колонки, перезаписати df лише з необхідними
                        if df.shape[1] > len(df_type):
                            df = df[df_type.keys()]

                        #Попередня обробка даних
                        df.drop('id', axis=1, inplace=True)


                        df = df_normal(df, df_minmax)
                        
                        #Прогнозування моделі та вивід результатів
                        df_file = model_predict(df, df_file, model)
                        st.dataframe(df_file.to_dict('records'))
                        # вивід точності якщо є колонка 'churn'
                        if 'churn' and 'pred_churn' in df_file.columns:
                            accuracy = accuracy_score(df_file['churn'], df_file['pred_churn'])
                            st.write(f'accuracy: {accuracy:.2%}')
                

    #Якщо обрано ввід з клавіатури
    elif input_type == 'Enter from the keyboard':
        # Створення полів вводу даних
        id = st.number_input('id:', step=1, min_value=1, key='id')
        # створення трьох колонок
        col11, col12, col13 = st.columns(3)
        with col11:
            st.radio('is_tv_subscriber', [0, 1], key='is_tv_subscriber')
        with col12:
            st.radio('is_movie_package_subscriber', [0, 1], key='is_movie_package_subscriber')
        with col13:
            st.number_input('subscription_age:', min_value=0.00, value=0.00, key='subscription_age')

        col21, col22, col23 = st.columns(3)
        with col21:
            st.number_input('bill_avg:', step=1, min_value=0, key='bill_avg')
        with col22:
            st.number_input('reamining_contract:', min_value=0.00, value=0.00, key='reamining_contract')
        with col23:
            st.number_input('service_failure_count:', step=1, min_value=0, key='service_failure_count')
        
        col31, col32, col33 = st.columns(3)
        with col31:
            st.number_input('download_avg:', min_value=0.00, value=0.00, key='download_avg')
        with col32:
            st.number_input('upload_avg:', min_value=0.00, value=0.00, key='upload_avg')
        with col33:
            st.number_input('download_over_limit:', step=1, min_value=0, key='download_over_limit')

        #Якщо кнопка для додавання даних натиснута, додати дані з полів
        if st.button("add a client"):
            #st.session_state в Streamlit використовується для зберігання стану між різними викликами функції streamlit run,
            #що дозволяє зберігати та змінювати значення змінних під час сесії користувача.
            #Якщо 'dataframe' не існує в st.session_state, створити st.session_state['dataframe']
            if 'dataframe' not in st.session_state:
                st.session_state['dataframe'] = pd.DataFrame(columns=[
                    'id', 'is_tv_subscriber', 'is_movie_package_subscriber',
                    'subscription_age', 'bill_avg', 'reamining_contract',
                    'service_failure_count', 'download_avg', 'upload_avg',
                    'download_over_limit'])
                
            # Якщо нове id вже існує, вивести попередження
            if id in st.session_state['dataframe']['id'].values: 
                st.error(f"id {id} already exists.")     
            else:
                #створення дата фрейму з введених даних
                new_data = add_data()
                #Додавання нового датафрейму (користувача) new_data до st.session_state['dataframe']
                st.session_state['dataframe'] = pd.concat([st.session_state['dataframe'], new_data], ignore_index=True)

        #Можливість видалення клієнта по id
        #поле для введення id клієнта для видалення
        del_id = st.number_input('deletable identifier:', step=1, min_value=0)
        #якщо кнопка "delete the client" натиснута
        if st.button("delete the client"):
            #Перевірити чи існує id та видалити клієнта
            if id in st.session_state['dataframe']['id'].values:
                #перезаписання у  st.session_state['dataframe'] всіх id які != del_id
                st.session_state['dataframe'] = st.session_state['dataframe'].loc[st.session_state['dataframe']['id'] != del_id]    
        
        #Вивід даних
        if 'dataframe' in st.session_state:
            st.write("Current data:")
            st.dataframe(st.session_state['dataframe'].to_dict('records'))

        #Якщо натиснута кнопка "predict"
        if st.button("predict"):
            try:
                #запис з st.session_state['dataframe'] у початкові дані введені з клавіатури
                df_input = st.session_state['dataframe']
                #Перезапис у датафрейм який буде змінюватися 
                df = df_input
                #перевірка на коректність вводу
                if data_validation(df, df_type):
                    #заміна всіх пустих значень на 0
                    df = df.fillna(0)
                    if df.shape[1] > len(df_type):
                        df = df[df_type.keys()]
                    #Попередня обробка даних
                    df.drop('id', axis=1, inplace=True)
                    #Нормалізація
                    df = df_normal(df, df_minmax)
                    #Обробка моделлю
                    df_input = model_predict(df, df_input, model)
                    st.dataframe(df_input.to_dict('records'))
            #Якщо st.session_state['dataframe'] не існує
            except:
                st.warning("Please add the data.")

if __name__ == "__main__":
    main()        