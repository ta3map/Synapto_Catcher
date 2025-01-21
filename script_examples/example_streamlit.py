import streamlit as st

# Настройка стилей с помощью Markdown и CSS
st.markdown("""
    <style>
        .main {
            background-color: #333333;
            padding: 20px;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
        }
        .stTextInput > div {
            border-radius: 10px;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #333;
            color: white;
            text-align: center;
            padding: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Заголовок приложения
st.title("Полный пример приложения на Streamlit")

# Текстовый ввод
st.subheader("Ввод данных")
name = st.text_input("Введите ваше имя:")
if name:
    st.write(f"Привет, {name}!")

# Чекбокс
st.subheader("Чекбокс")
show_greeting = st.checkbox("Показать приветствие")
if show_greeting:
    st.write("Привет! Рад вас видеть.")

# Радиокнопки
st.subheader("Выбор пола")
gender = st.radio("Укажите ваш пол:", ("Мужчина", "Женщина", "Не указывать"))
if gender:
    st.write(f"Вы выбрали: {gender}")

# Выпадающий список
st.subheader("Выбор возраста")
age_group = st.selectbox("Выберите возрастную группу:", 
                         ("< 18", "18-30", "31-50", "50+"))
st.write(f"Вы выбрали: {age_group}")

# Ползунок
st.subheader("Настройка яркости")
brightness = st.slider("Выберите уровень яркости", 0, 100, 50)
st.write(f"Текущая яркость: {brightness}%")

# Мультивыбор
st.subheader("Мультивыбор увлечений")
hobbies = st.multiselect(
    "Выберите ваши увлечения:",
    ["Чтение", "Спорт", "Музыка", "Программирование", "Путешествия"]
)
if hobbies:
    st.write(f"Вы выбрали следующие увлечения: {', '.join(hobbies)}")

# Цветовой выбор
st.subheader("Выбор цвета")
color = st.color_picker("Выберите цвет фона", "#ffffff")
st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;'>Это ваш выбранный цвет</div>", unsafe_allow_html=True)

# Загрузка файлов
st.subheader("Загрузка файла")
uploaded_file = st.file_uploader("Загрузите файл", type=["txt", "csv", "jpg", "png"])
if uploaded_file is not None:
    st.write("Файл загружен успешно!")

# Диаграмма
st.subheader("Пример диаграммы")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'x': np.arange(1, 11),
    'y': np.random.randint(1, 100, size=10)
})

fig, ax = plt.subplots()
ax.plot(data['x'], data['y'], label="Random Data", marker='o')
ax.set_title("Пример линейного графика")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()

st.pyplot(fig)

# Кнопка
st.subheader("Кнопка для тестирования")
if st.button("Нажми меня"):
    st.write("Кнопка нажата!")

# Сообщение об успехе
if st.button("Успешное действие"):
    st.success("Действие завершено успешно!")

# Прогресс-бар
st.subheader("Прогресс-бар")
import time

progress = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress.progress(i+1)

# Сообщение в футере
st.markdown("""
    <div class="footer">
        <p>Создано с любовью с использованием Streamlit!</p>
    </div>
""", unsafe_allow_html=True)
