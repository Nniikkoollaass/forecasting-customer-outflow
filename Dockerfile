# Указываем базовый образ
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл requirements.txt в рабочую директорию
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной исходный код в рабочую директорию
COPY . .

# Указываем команду для запуска приложения
CMD ["streamlit", "run", "app.py"]


