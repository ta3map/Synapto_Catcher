import subprocess
import sys
import os
import streamlit as st
import json

def normalize_paths(path):
    """
    Функция для нормализации пути с учетом регистра и существования файла.
    Возвращает правильный путь с учетом регистра символов.
    """
    if os.path.exists(path):
        return os.path.normpath(path)  # Не realpath, чтобы сохранить букву диска
    else:
        raise FileNotFoundError(f"Path does not exist: {path}")

def wrap_path_if_needed(path):
    """
    Заключает путь в кавычки, если в нем есть пробелы.
    """
    if " " in path:
        return f'"{path}"'
    return path

def run_pyinstaller_as_module(pyinstaller_args):
    """
    Запускает PyInstaller как модуль Python и выводит результат в реальном времени.
    :param pyinstaller_args: Список аргументов для PyInstaller.
    """
    try:
        # Запускаем PyInstaller как модуль через текущую интерпретацию Python
        command = [sys.executable, '-m', 'PyInstaller'] + pyinstaller_args

        # Выводим команду для отладки
        st.write("Executing PyInstaller command as Python module:")
        st.code(" ".join(command))

        # Открываем subprocess и используем Popen для получения вывода в реальном времени
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Читаем и выводим stdout и stderr в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                st.text(output.strip())  # Выводим результат stdout в Streamlit

        # Проверка на ошибки
        stderr = process.stderr.read()
        if stderr:
            st.error(stderr.strip())

        # Проверяем код возврата процесса
        return_code = process.poll()
        if return_code == 0:
            st.success("PyInstaller executed successfully.")
        else:
            st.error(f"PyInstaller failed with return code {return_code}.")

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Streamlit interface
st.title("PyInstaller GUI")

# Initialize default values
script_file = ""
icon_file = ""
splash_screen_file = ""
onefile = True
console = True
app_name = ""
clean_build = False
strip = False
noupx = False
disable_windowed_traceback = False
uac_admin = False
additional_files = {}
advanced_options = ""

# Initialize empty command list to avoid NameError
command = []

# Load settings from JSON file
import_file = st.file_uploader("Import Settings from JSON", type=["json"])
if import_file:
    try:
        # Загружаем JSON с помощью json.load и streamlit's file_uploader
        settings = json.load(import_file)

        # Обновляем переменные после импорта JSON
        for opt in settings["pyinstallerOptions"]:
            if opt["optionDest"] == "filenames":
                script_file = opt["value"]
            if opt["optionDest"] == "onefile":
                onefile = opt["value"]  # Обновляем переменную onefile
            if opt["optionDest"] == "console":
                console = opt["value"]  # Обновляем переменную console
            if opt["optionDest"] == "icon_file":
                icon_file = opt["value"]
            if opt["optionDest"] == "name":
                app_name = opt["value"]
            if opt["optionDest"] == "clean_build":
                clean_build = opt["value"]  # Обновляем clean_build
            if opt["optionDest"] == "splash":
                splash_screen_file = opt["value"]
            if opt["optionDest"] == "strip":
                strip = opt["value"]
            if opt["optionDest"] == "noupx":
                noupx = opt["value"]
            if opt["optionDest"] == "disable_windowed_traceback":
                disable_windowed_traceback = opt["value"]
            if opt["optionDest"] == "uac_admin":
                uac_admin = opt["value"]
            if opt["optionDest"] == "datas":
                # Разделяем путь на источник и папку назначения, используя двоеточие для Windows
                try:
                    file, folder = opt["value"].split(";")
                    additional_files[normalize_paths(file)] = normalize_paths(folder)
                except ValueError:
                    st.error(f"Error parsing 'datas' field: {opt['value']}")

        # Пересобираем строку команды сразу после обновления переменных
        command = ["--noconfirm"]
        command.append("--onefile" if onefile else "--onedir")
        command.append("--console" if console else "--windowed")

        # Добавляем остальные параметры
        if icon_file:
            command.append(f'--icon={wrap_path_if_needed(normalize_paths(icon_file))}')
        if app_name:
            # Заключаем имя приложения в кавычки
            command.append(f'--name={wrap_path_if_needed(app_name)}')  
        if clean_build:
            command.append("--clean")  # Добавляем --clean, если флаг clean_build выбран
        if splash_screen_file:
            command.append(f'--splash={wrap_path_if_needed(normalize_paths(splash_screen_file))}')
        for file, folder in additional_files.items():
            command.append(f'--add-data={wrap_path_if_needed(normalize_paths(file))}:{wrap_path_if_needed(normalize_paths(folder))}')  # Используем двоеточие для разделения
        command.append(wrap_path_if_needed(normalize_paths(script_file)))

        # Отображаем собранную команду
        st.code(" ".join(command))

        st.success("Settings imported and command updated successfully!")
    except Exception as e:
        st.error(f"Failed to import settings: {e}")

# Script and icon fields
script_file = st.text_input("Python script path", value=script_file)
icon_file = st.text_input("Icon path (.ico)", value=icon_file)
splash_screen_file = st.text_input("Splash screen path", value=splash_screen_file)

# Compilation settings
st.header("Compilation Settings")

# Onefile or onedir
onefile = st.radio("Packaging Mode", ["One File", "One Directory"], index=0 if onefile else 1)

# Console or windowed application
console = st.radio("Console Window", ["Console Based", "Window Based (No Console)"], index=0 if console else 1)

# Additional files
st.header("Additional Files")
num_files = st.number_input("How many additional files to add?", min_value=0, value=len(additional_files), step=1)

for i, (file, folder) in enumerate(additional_files.items()):
    file_input = st.text_input(f"File {i+1}", value=file)
    folder_input = st.text_input(f"Folder for File {i+1}", value=folder)
    additional_files[normalize_paths(file_input)] = normalize_paths(folder_input) if folder_input else "."

# Application name and options
app_name = st.text_input("Executable Name", value=app_name)
clean_build = st.checkbox("Clean Build", value=clean_build)  # Чекбокс для clean_build
strip = st.checkbox("Strip Debug Info", value=strip)
noupx = st.checkbox("Disable UPX Compression", value=noupx)
disable_windowed_traceback = st.checkbox("Disable Windowed Traceback", value=disable_windowed_traceback)
uac_admin = st.checkbox("Request UAC Admin Privileges", value=uac_admin)

# Advanced options
st.header("Advanced Options")
advanced_options = st.text_input("Additional PyInstaller Options", value=advanced_options)

# Export settings
st.header("Export/Import Settings")
export_filename = st.text_input("Export filename", value="pyinstaller_settings.json")
if st.button("Export Settings"):
    export_settings(export_filename, script_file, onefile, console, icon_file, additional_files, advanced_options, app_name,
                    splash_screen_file, clean_build, strip, noupx, disable_windowed_traceback, uac_admin)

# Run PyInstaller button
if st.button("Convert .py to .exe"):
    if script_file:
        st.write("Starting conversion...")

        # Запускаем PyInstaller как модуль через текущий интерпретатор
        run_pyinstaller_as_module(command)  # Используем ту же команду, что и выше
    else:
        st.write("Please provide a Python script path.")

# Визуализация строки команды для отладки
st.write("Generated PyInstaller command preview:")
st.code(" ".join(command))  # Отображаем команду
