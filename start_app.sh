#!/bin/bash

# Скрипт для запуска Synapto Catcher
# Активирует виртуальное окружение и запускает app.py

# Функция для вывода ошибок
error_exit() {
    echo "ОШИБКА: $1" >&2
    echo "Нажмите Enter для выхода..."
    read
    exit 1
}

# Функция для паузы
pause() {
    echo "Нажмите Enter для продолжения..."
    read
}

# Получаем директорию скрипта
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Переходим в директорию проекта
cd "$SCRIPT_DIR" || error_exit "Не удалось перейти в директорию проекта"

echo "Рабочая директория: $SCRIPT_DIR"

# Проверяем существование виртуального окружения
if [ ! -d "venv" ]; then
    echo "Виртуальное окружение не найдено. Создаю..."
    python3 -m venv venv || error_exit "Не удалось создать виртуальное окружение"
fi

# Активируем виртуальное окружение
echo "Активирую виртуальное окружение..."
source venv/bin/activate || error_exit "Не удалось активировать виртуальное окружение"

# Проверяем установлены ли зависимости
if [ ! -d venv/lib/python*/site-packages/numpy* ]; then
    echo "Устанавливаю зависимости..."
    pip install --upgrade pip || error_exit "Не удалось обновить pip"
    
    # Используем минимальный файл requirements
    if [ -f "requirements_clean.txt" ]; then
        echo "Устанавливаю пакеты из requirements_clean.txt..."
        pip install -r requirements_clean.txt || error_exit "Не удалось установить зависимости из requirements_clean.txt"
    elif [ -f "requirements.txt" ]; then
        echo "Устанавливаю пакеты из requirements.txt..."
        pip install -r requirements.txt || error_exit "Не удалось установить зависимости из requirements.txt"
    else
        error_exit "Не найден файл requirements"
    fi
fi

# Проверяем существование app.py
if [ ! -f "app.py" ]; then
    error_exit "Файл app.py не найден"
fi

# Запускаем приложение
echo "Запускаю Synapto Catcher..."
python app.py

# Проверяем код завершения приложения
if [ $? -ne 0 ]; then
    echo "Приложение завершилось с ошибкой"
    pause
fi

# Деактивируем окружение после завершения
deactivate

echo "Программа завершена."
pause 