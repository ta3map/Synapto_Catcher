import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import subprocess

from graphical_processor import display_thumbnails, process_czi_image

# Тестовая функция для одиночного клика
def test_single_click(image_path):
    print(f"Single click on: {image_path}")

# Функция для открытия изображения в системе (пример функции для двойного клика)
def open_image(image_path):
    print(f"Open the file: {image_path}")
    try:
        if os.name == 'nt':  # Windows
            os.startfile(image_path)
        elif os.name == 'posix':  # macOS или Linux
            subprocess.call(('open' if sys.platform == 'darwin' else 'xdg-open', image_path))
    except Exception as e:
        messagebox.showerror("Error", f"Failed to open file: {e}")

            
# Тестовая кастомная функция для открытия изображения через OpenCV
def custom_open_image_with_opencv(image_path):
    # Открываем изображение с помощью OpenCV
    img_cv = cv2.imread(image_path)
    
    if img_cv is None:
        raise ValueError(f"Could not open image: {image_path}")

    # Преобразуем цвет с BGR на RGB
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

    # Преобразуем в формат PIL
    img_pil = Image.fromarray(img_rgb)
    
    return img_pil


# Function for selecting a file and viewing the folder contents
def load_images():
    file_selected = filedialog.askopenfilename(title="Select a file in the folder", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if not file_selected:
        messagebox.showwarning("Error", "No file selected")
        return

    # Get the folder path from the selected file
    folder_selected = os.path.dirname(file_selected)

    # Fetch all image files from the folder
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.lower().endswith(valid_extensions)]

    if not images:
        messagebox.showinfo("Info", "No images found in the folder.")
        return

    # Code to handle and display images as thumbnails
    display_thumbnails(display_frame, images, max_per_page=4)


# Создаем основное окно
root = tk.Tk()
root.title("Image Gallery")
root.geometry("600x400")

# Фрейм для отображения миниатюр
display_frame = tk.Frame(root)
display_frame.pack(fill=tk.BOTH, expand=True)

# Кнопка для выбора папки с изображениями
select_folder_button = tk.Button(root, text="Select folder with images", command=load_images)
select_folder_button.pack(side=tk.TOP, pady=10)

# Запуск основного цикла
root.mainloop()



