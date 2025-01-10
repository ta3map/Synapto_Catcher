import cv2
import numpy as np
from matplotlib import cm
import pandas as pd
from czifile import CziFile
from scipy.ndimage import zoom
from io import BytesIO
import hashlib
import tempfile
import threading
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageOps
from functools import partial
import os
import subprocess
import cv2
import math
import keyboard
from tkinter import Listbox, Canvas
import json
from tkinter import filedialog, messagebox, PhotoImage, Toplevel, scrolledtext, StringVar, BooleanVar, Tk, simpledialog 
from tkinter.ttk import Button, Label, Entry, OptionMenu, Style, Checkbutton, Frame, Progressbar, Scale, Scrollbar
from readlif.reader import LifFile


class ThumbnailViewer:
    def __init__(self, parent, images, 
                 comments=None, image_ids=None,
                 replaced_image_names = None,
                 on_single_click=lambda path: None, 
                 on_double_click=lambda path: None, 
                 on_selection_change=lambda selected_ids: None,
                 open_image_func=Image.open, 
                 max_per_page=None, width=500, height=150):
        
        self.parent = parent
        self.images = images
        self.comments = comments
        self.image_ids = image_ids
        self.on_selection_change = on_selection_change
        
        self.replaced_image_names = replaced_image_names
        
        self.on_single_click = on_single_click
        self.on_double_click = on_double_click
        self.open_image_func = open_image_func
        self.max_per_page = max_per_page
        self.width = width
        self.height = height
        
        self.progress_window = None
        self.progress_bar = None
        self.progress_label = None
        self.empty_comments = False

        self.cancel_loading = False

        # Переменные для постраничной навигации
        self.current_page = 0
        if self.max_per_page is None:
            self.max_per_page = len(self.images)

        self.TEMP_FILE = os.path.join(tempfile.gettempdir(), 'agr_thumbnails')
        if not os.path.exists(self.TEMP_FILE):
            os.makedirs(self.TEMP_FILE)

        # Переменные для перетаскивания и клика
        self.start_x = 0
        self.start_y = 0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        self.last_clicked_thumbnail = None  # Для хранения последней нажатой миниатюры
        self.selected_images = set()  # Набор путей выбранных изображений

        self._setup_ui()
        self._load_page_images(0)
        
    def display_selected_ids(self):
        """Отображаем выделенные ID в текстовом поле в порядке следования изображений."""
        if self.image_ids is None:
            self.selected_ids_label.pack_forget()  # Скрываем текстовое поле, если ID не переданы
            return

        if not self.selected_images:
            self.selected_ids_label.pack_forget()  # Скрываем текстовое поле, если нет выделений
            return

        # Создаем словарь для быстрого поиска ID по изображению
        image_to_id = {img: img_id for img, img_id in zip(self.images, self.image_ids)}

        # Собираем список ID для выделенных изображений
        selected_ids_num = [image_to_id[img] for img in self.selected_images if img in image_to_id]
        selected_ids_num.sort(key=lambda x: list(self.image_ids).index(x))


        # Преобразуем все ID в строки
        selected_ids = [str(id) for id in selected_ids_num]

        # Формируем сокращенный вывод
        max_displayed_ids = 30  # Разрешенное количество для отображения

        if len(selected_ids) > max_displayed_ids:
            # Количество чисел, которые будут показываться с начала и с конца
            start_count = (max_displayed_ids // 2)  # Числа с начала
            end_count = (max_displayed_ids // 2)    # Числа с конца

            # Формирование списка с началом, троеточиями и концом
            displayed_ids = ", ".join(selected_ids[:start_count]) + " ... " + ", ".join(selected_ids[-end_count:])
        else:
            displayed_ids = ", ".join(selected_ids)

        # Обновляем текстовое поле
        self.selected_ids_label.config(text=f"Selected IDs: {displayed_ids}")
        self.selected_ids_label.pack(side=tk.BOTTOM, fill=tk.X)  # Отображаем текстовое поле
        self.on_selection_change(selected_ids_num)


    def _setup_ui(self):
        # Создаем фрейм для миниатюр
        self.thumbnail_frame = Frame(self.parent)
        self.thumbnail_frame.pack(fill=tk.BOTH, expand=True)

        # Создаем canvas для размещения миниатюр и scrollbar
        self.canvas = Canvas(self.thumbnail_frame, width=self.width, height=self.height, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.thumbnail_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.config(xscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.thumbnail_inner_frame = Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.thumbnail_inner_frame, anchor="nw")
        self.thumbnail_inner_frame.bind("<MouseWheel>", self.on_mouse_wheel)

        self.images = [image for image in self.images if os.path.exists(image)]
        
        # Если комментарии переданы, проверяем их длину
        if self.comments is None:
            self.comments = ['' for _ in self.images]  # Если комментариев нет, заполняем пустыми строками
            self.empty_comments = True
        else:
            if len(self.comments) != len(self.images):
                raise ValueError("Количество комментариев должно соответствовать количеству изображений.")

        # Кнопки навигации
        self.prev_button = Button(self.thumbnail_frame, text="🡸", command=self.show_previous_page)
        self.next_button = Button(self.thumbnail_frame, text="🡺", command=self.show_next_page)

        self.prev_button.pack(side=tk.LEFT, padx=5)
        self.next_button.pack(side=tk.RIGHT, padx=5)

        # Добавим поле для отображения выбранных ID
        self.selected_ids_label = Label(self.thumbnail_frame, text="")
        self.selected_ids_label.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        # Добавляем кнопку для выделения всех изображений
        self.select_all_button = Button(self.thumbnail_frame, text="Select all", command=self.select_all_images)
        self.select_all_button.pack(side=tk.LEFT, padx=5)
    
        # Получаем корневое окно (Toplevel), содержащее этот виджет
        self.root = self.parent.winfo_toplevel()

        # Связываем события клавиш с этим окном
        self.root.bind("<Control-a>", self.on_ctrl_a)
        self.root.bind("<Control-A>", self.on_ctrl_a)
        self.root.bind("<Control-Shift-a>", self.on_ctrl_shift_a)
        self.root.bind("<Control-Shift-A>", self.on_ctrl_shift_a)

    def get_thumbnail_filename(self, image_path):
        # Получаем время модификации файла в виде целого числа
        modification_time = int(os.path.getmtime(image_path))
        # Генерируем хэш от пути
        base_hash = hashlib.md5(image_path.encode()).hexdigest()
        # Склеиваем хэш и время модификации, чтобы получить уникальное имя
        file_name = f"{base_hash}_{modification_time}.png"
        return os.path.join(self.TEMP_FILE, file_name)

    def clear_thumbnails(self):
        try:
            if self.thumbnail_inner_frame.winfo_exists():  # Проверяем, существует ли фрейм
                for widget in self.thumbnail_inner_frame.winfo_children():
                    widget.destroy()  # Удаляем все дочерние виджеты
        except tk.TclError as e:
            print(f"Error during clearing thumbnails: {e}")

    def add_comment_to_image(self, image, comment):
        """Наносим комментарий поверх изображения с черным фоном, жирным шрифтом и размером 14."""
        # Копируем изображение, чтобы не изменять оригинал
        
            # Преобразуем изображение в RGB, если оно в другом формате
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)

        # Определяем шрифт, пытаемся загрузить Arial с размером 14 и жирным шрифтом
        try:
            font = ImageFont.truetype("arialbd.ttf", 14)  # Используем жирный Arial
        except IOError:
            font = ImageFont.load_default()

        # Используем font.getbbox() для расчета размера текста
        text_bbox = draw.textbbox((0, 0), comment, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Определяем координаты и размер черного фона (немного больше текста)
        padding = 5
        text_x, text_y = 10, 10  # Положение текста
        background_x1 = text_x - padding
        background_y1 = text_y - padding
        background_x2 = text_x + text_width + padding
        background_y2 = text_y + text_height + padding
        
        # Рисуем черный прямоугольник под текстом
        draw.rectangle([background_x1, background_y1, background_x2, background_y2], fill=(0, 0, 0))

        # Цвет текста - белый
        text_color = (255, 255, 255)
        
        # Наносим текст поверх черного фона
        draw.text((text_x, text_y), comment, font=font, fill=text_color)

        return img_copy

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            self.canvas.xview_scroll(-1, "units")  # Прокрутка влево
        else:
            self.canvas.xview_scroll(1, "units")   # Прокрутка вправо

    def on_mouse_press(self, event):        
        """Запоминаем начальную позицию мыши при нажатии."""
        self.start_x = event.x
        self.start_y = event.y
        self.canvas_offset_x = self.canvas.canvasx(0)
        self.canvas_offset_y = self.canvas.canvasy(0)

    def on_mouse_drag(self, event):
        """Изменяем положение видимой области холста по движению мыши."""
        delta_x = self.start_x - event.x
        self.canvas.xview_moveto((self.canvas_offset_x + delta_x) / self.canvas.bbox("all")[2])

    def on_mouse_release(self, event, img_path, thumbnail_label):
        # Получаем цвет подсветки через ThemeManager
        theme_manager = ThemeManager()
        highlight_color = theme_manager.get_theme_highlight() or "blue"  # Используем цвет подсветки или синий по умолчанию
        
        """Определяем, было ли это кликом или перетаскиванием."""
        delta_x = event.x - self.start_x
        delta_y = event.y - self.start_y
        movement = (delta_x**2 + delta_y**2)**0.5

        if movement < 5:  # Порог для определения клика
            # Проверяем, зажата ли клавиша Ctrl
            if event.state & 0x0004:  # Проверяем, зажата ли клавиша Ctrl
                # Множественное выделение
                if img_path in self.selected_images:
                    # Если изображение уже выбрано, снимаем выделение
                    self.selected_images.remove(img_path)
                    thumbnail_label.config(borderwidth=0, relief="flat")
                else:
                    # Добавляем изображение в выбранные
                    self.selected_images.add(img_path)
                    thumbnail_label.config(borderwidth=4, relief="flat", background = highlight_color)                
            else:
                # Одиночное выделение
                # Сбрасываем выделение предыдущих изображений
                for widget in self.thumbnail_inner_frame.winfo_children():
                    lbl = widget.children.get('!label')
                    if lbl:
                        lbl.config(borderwidth=0, relief="flat")
                self.selected_images.clear()

                # Выделяем текущую миниатюру
                self.selected_images.add(img_path)
                thumbnail_label.config(borderwidth=4, relief="flat", background = highlight_color)
                self.last_clicked_thumbnail = thumbnail_label
                
            self.display_selected_ids()

            # Вызываем функцию on_single_click
            self.on_single_click(img_path)
        else:
            # Это перетаскивание, ничего не делаем
            pass

    def on_ctrl_a(self, event):
        """Обрабатываем нажатие Ctrl+A для выделения всех видимых изображений."""
        focus_widget = self.root.focus_get()
        if focus_widget and str(focus_widget).startswith(str(self.root)):
            self.select_all_visible()
        return "break"  # Останавливаем дальнейшую обработку события

    def on_ctrl_shift_a(self, event):
        """Обрабатываем нажатие Ctrl+Shift+A для выделения всех изображений."""
        focus_widget = self.root.focus_get()
        if focus_widget and str(focus_widget).startswith(str(self.root)):
            self.select_all_images()
        return "break"  # Останавливаем дальнейшую обработку события


    def select_all_visible(self):
        """Выделяем все видимые изображения на текущей странице с использованием цвета подсветки из темы."""
        # Получаем цвет подсветки через ThemeManager
        theme_manager = ThemeManager()
        highlight_color = theme_manager.get_theme_highlight() or "blue"  # Используем цвет подсветки или синий по умолчанию

        for widget in self.thumbnail_inner_frame.winfo_children():
            thumbnail_label = widget.children.get('!label')
            if thumbnail_label:
                img_path = thumbnail_label.img_path
                self.selected_images.add(img_path)
                thumbnail_label.config(borderwidth=4, relief="flat", background=highlight_color)
                self.display_selected_ids()

    def select_all_images(self):
        """Выделяем все изображения на всех страницах."""
        self.selected_images = set(self.images)  # Выбираем все изображения
        self.select_all_visible()
        self.display_selected_ids()
        #self.update_thumbnails_selection()  # Обновляем отображение на текущей странице

    def update_thumbnails_selection(self):        
        # Получаем цвет подсветки через ThemeManager
        theme_manager = ThemeManager()
        highlight_color = theme_manager.get_theme_highlight() or "blue"  # Используем цвет подсветки или синий по умолчанию
        """Обновляем отображение выделения на текущей странице."""
        for widget in self.thumbnail_inner_frame.winfo_children():
            thumbnail_label = widget.children.get('!label')
            if thumbnail_label:
                img_path = thumbnail_label.img_path
                if img_path in self.selected_images:
                    thumbnail_label.config(borderwidth=4, relief="flat", background = highlight_color)                    
                else:
                    thumbnail_label.config(borderwidth=0, relief="flat")
                self.display_selected_ids()

    def create_progress_window(self, max_value):
        """
        Create a progress window and a progress bar.
        """
        self.progress_window = initialize_window(self.parent, "Loading Images", 400, 200, icon_path=icon_path)
        
        self.progress_label = Label(self.progress_window, text="Loading images, please wait...")
        self.progress_label.pack(padx=20, pady=10)

        self.progress_bar = Progressbar(self.progress_window, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(padx=20, pady=10)
        self.progress_bar["maximum"] = max_value

        cancel_button = Button(self.progress_window, text="Cancel", command=self.cancel_progress)
        cancel_button.pack(pady=10)

    def close_progress_window(self):
        if self.progress_window is not None:
            self.progress_window.destroy()
            self.progress_window = None

    def cancel_progress(self):
        self.cancel_loading = True
        self.close_progress_window()

    def _load_page_images(self, start_index):
        threading.Thread(target=self._load_images, args=(start_index,)).start()

    def _load_images(self, start_index):
        self.cancel_loading = False
        self.clear_thumbnails()
        
        # Получаем цвет подсветки через ThemeManager
        theme_manager = ThemeManager()
        highlight_color = theme_manager.get_theme_highlight() or "blue"  # Используем цвет подсветки или синий по умолчанию
        
        # Устанавливаем количество изображений для текущей страницы
        end_index = min(start_index + self.max_per_page, len(self.images))
        page_image_count = end_index - start_index  # Количество изображений на текущей странице

        self.create_progress_window(page_image_count)

        for idx_in_page, img_path in enumerate(self.images[start_index:end_index], start=start_index):
            if self.cancel_loading:
                break

            # Проверяем, существует ли thumbnail_inner_frame перед созданием виджетов
            if not self.thumbnail_inner_frame.winfo_exists():
                break

            thumbnail_path = self.get_thumbnail_filename(img_path)
            if os.path.exists(thumbnail_path):
                img = Image.open(thumbnail_path)
            else:
                img = self.open_image_func(img_path)
                img.thumbnail((150, 150))
                img.save(thumbnail_path, format='PNG')

            # Добавляем комментарий к изображению
            if not self.empty_comments:
                img = self.add_comment_to_image(img, self.comments[idx_in_page])

            img_tk = ImageTk.PhotoImage(img)

            thumbnail_container = Frame(self.thumbnail_inner_frame)
            thumbnail_label = Label(thumbnail_container, image=img_tk)
            thumbnail_label.image = img_tk
            thumbnail_label.img_path = img_path  # Сохраняем путь в атрибуте
            thumbnail_label.pack(side=tk.TOP, padx=5, pady=5)

            # Проверяем, выбрано ли изображение
            if img_path in self.selected_images:
                thumbnail_label.config(borderwidth=4, relief="flat", background = highlight_color)            
            else:
                thumbnail_label.config(borderwidth=0, relief="flat")
            self.display_selected_ids()

            if not self.replaced_image_names: # заменяем названия файлов
                file_name = os.path.basename(img_path)
            else:
                file_name = self.replaced_image_names[start_index+idx_in_page]
                
            file_label = Label(thumbnail_container, text=file_name, font=("Arial", 10), wraplength=100)
            file_label.pack(side=tk.TOP)

            thumbnail_container.pack(side=tk.LEFT, padx=5, pady=5)

            # Привязываем события перетаскивания и клика к миниатюрам
            thumbnail_label.bind("<ButtonPress-1>", self.on_mouse_press)
            thumbnail_label.bind("<B1-Motion>", self.on_mouse_drag)
            thumbnail_label.bind("<ButtonRelease-1>", partial(self.on_mouse_release, img_path=img_path, thumbnail_label=thumbnail_label))
            thumbnail_label.bind("<Double-Button-1>", lambda event, path=img_path: self.on_double_click(path))

            thumbnail_container.bind("<MouseWheel>", self.on_mouse_wheel)
            thumbnail_label.bind("<MouseWheel>", self.on_mouse_wheel)
            file_label.bind("<MouseWheel>", self.on_mouse_wheel)

            # Обновляем прогресс-бар
            if self.progress_bar:
                self.progress_bar["value"] = idx_in_page - start_index + 1  # Обновляем прогресс для изображений на текущей странице
                self.progress_label.config(text=f"Loading image {idx_in_page - start_index + 1} of {page_image_count}")
                self.parent.update_idletasks()  # Обновляем интерфейс

        self.close_progress_window()

        # Проверяем, существует ли canvas перед вызовом bbox
        if self.canvas.winfo_exists():
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            
        self.thumbnail_inner_frame.update_idletasks()

        self.update_buttons()
        #self.update_thumbnails_selection()

    def show_next_page(self):
        if (self.current_page + 1) * self.max_per_page < len(self.images):
            self.current_page += 1

            self._load_page_images(self.current_page * self.max_per_page)

            # Перемещаем слайдер в начало, чтобы показывать новые миниатюры с начала страницы
            self.canvas.xview_moveto(0)  # Перемещаем слайдер в начало

    def show_previous_page(self):
        if self.current_page > 0:
            self.current_page -= 1

            self._load_page_images(self.current_page * self.max_per_page)

            # Перемещаем слайдер в конец, чтобы показывать конец предыдущей страницы
            self.canvas.xview_moveto(1)  # Перемещаем слайдер в конец

    def update_buttons(self):
        # Проверяем существование кнопки перед обновлением ее состояния
        if self.prev_button.winfo_exists():
            if self.current_page == 0:
                self.prev_button.config(state=tk.DISABLED)
            else:
                self.prev_button.config(state=tk.NORMAL)

        if self.next_button.winfo_exists():
            if (self.current_page + 1) * self.max_per_page >= len(self.images):
                self.next_button.config(state=tk.DISABLED)
            else:
                self.next_button.config(state=tk.NORMAL)

        if len(self.images) <= self.max_per_page:
            if self.prev_button.winfo_exists():
                self.prev_button.config(state=tk.DISABLED)
            if self.next_button.winfo_exists():
                self.next_button.config(state=tk.DISABLED)

    def destroy(self):
        """Метод для очистки виджетов и отвязки обработчиков событий."""
        # Отвязываем обработчики событий
        self.root.unbind("<Control-a>")
        self.root.unbind("<Control-A>")
        self.root.unbind("<Control-Shift-a>")
        self.root.unbind("<Control-Shift-A>")

        # Уничтожаем все виджеты
        self.thumbnail_frame.destroy()



        
def create_excel_snapshot_to_image(excel_file, sheet_name=None, rows=10, cols=5):
    # Чтение Excel файла
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Если файл содержит несколько листов, выбираем первый лист
    if isinstance(df, dict):
        # Если указано имя листа, используем его
        if sheet_name is not None:
            df = df[sheet_name]
        else:
            # Используем первый лист
            df = list(df.values())[0]
    
    # Обрезаем до заданного количества строк и столбцов, добавляем доп. колонку с троеточиями
    df_snapshot = df.iloc[:rows, :cols]
    df_snapshot["..."] = "..."  # Добавляем колонку с троеточиями

    # Заменяем NaN на пустые строки
    df_snapshot = df_snapshot.fillna("")

    # Добавляем дополнительную строку с троеточиями
    df_snapshot.loc[len(df_snapshot)] = ["..." for _ in range(cols + 1)]

    # Настройка шрифта
    font = ImageFont.truetype("arial.ttf", 16)  # Arial с размером 16
    
    # Размеры для отрисовки
    line_spacing = 30  # Отступ между строками
    column_spacing = 150  # Ширина одного столбца
    num_cols = df_snapshot.shape[1]  # Количество столбцов
    
    text_height = (rows + 2) * line_spacing  # Высота текста с учетом отступов и заголовка
    text_width = column_spacing * num_cols  # Ширина всего изображения
    
    # Создание изображения
    image = Image.new('RGB', (text_width, text_height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Координаты для начала отрисовки текста
    x_start, y_start = 10, 10
    text_offset = 5  # Отступ текста от вертикальных линий
    
    # Цвета для чередования строк
    row_colors = ["white", "#D3D3D3"]  # Белый и светло-серый
    
    # Отрисовка заголовков столбцов
    header_y = y_start
    header_color = "#A9A9A9"  # Цвет фона заголовка
    
    # Отрисовка фона для заголовков
    draw.rectangle([0, header_y, text_width, header_y + line_spacing], fill=header_color)
    
    # Отрисовка заголовков с отступами и вертикальными разделителями
    for j, col_name in enumerate(df_snapshot.columns):
        x = x_start + j * column_spacing
        draw.text((x + text_offset, header_y), str(col_name), font=font, fill='black')
    
    # Отрисовка строк данных с чередованием фона
    for i, row in enumerate(df_snapshot.itertuples(index=False, name=None)):
        y = y_start + (i + 1) * line_spacing  # Сдвиг вниз после заголовка
        row_color = row_colors[i % 2]
        
        # Отрисовываем фон для строки
        draw.rectangle([0, y, text_width, y + line_spacing], fill=row_color)
        
        # Отрисовка каждой ячейки в строке с фоном, соответствующим строке, только если ячейка непустая
        for j, cell in enumerate(row):
            x = x_start + j * column_spacing
            
            if cell != "":  # Если ячейка непустая, рисуем фон для текста
                draw.rectangle([x, y, x + column_spacing, y + line_spacing], fill=row_color)
            
            # Отрисовка текста с отступом, если ячейка непустая
            draw.text((x + text_offset, y), str(cell), font=font, fill='black')
    
    # Отрисовка вертикальных линий после текста для их отображения поверх
    for j in range(num_cols + 1):
        x = x_start + j * column_spacing
        draw.line([(x, y_start), (x, text_height)], fill='black', width=2)
    
    # Конвертация изображения в BytesIO для вывода
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    return Image.open(img_byte_arr)


def normalize_channel(channel_data):
    """Нормирует данные канала в диапазон от 0 до 255"""
    channel_min = channel_data.min()
    channel_max = channel_data.max()
    if channel_max > channel_min:
        normalized_data = (channel_data - channel_min) / (channel_max - channel_min) * 255
    else:
        normalized_data = np.zeros_like(channel_data)
    return normalized_data.astype(np.uint8)

def create_combined_image(image_data, channels, slices, scale, max_channel, max_slice):
    """Создает RGB-изображение из заданных каналов и слайдов с опциональным масштабированием"""
    
    # Фильтруем только валидные каналы и срезы
    valid_channels = [ch for ch in channels if ch < max_channel]  # Игнорируем каналы, которые out of bounds
    valid_slices = [sl for sl in slices if sl < max_slice]        # Игнорируем срезы, которые out of bounds

    sample_slices = []
    
    # Получаем данные по каждому каналу через максимумы по валидным срезам
    for ch in valid_channels:
        sample_slice = np.max(image_data[0, 0, ch, valid_slices, :, :, 0], axis=0)
        sample_slices.append(sample_slice)
    
    # Если scale не равен 1, уменьшаем изображение
    if scale != 1.0:
        sample_slices = [zoom(slice_data, (scale, scale), order=1) for slice_data in sample_slices]
    
    # Инициализируем пустое RGB изображение
    combined_image = np.zeros((*sample_slices[0].shape, 3), dtype='uint8')
    
    # Выбираем цвета для каналов (RGB и дополнительные)
    colormap = cm.get_cmap('rainbow', len(valid_channels))  # Используем колорбар для распределения цветов по каналам
    
    # Проходим по каналам и добавляем их в RGB-изображение
    for i, sample_slice in enumerate(sample_slices):
        normalized_slice = normalize_channel(sample_slice)
        color = np.array(colormap(i)[:3]) * 255  # Получаем RGB цвет из колорбара
        for j in range(3):  # RGB компоненты
            combined_image[:, :, j] += (normalized_slice * color[j] / 255).astype(np.uint8)

    return combined_image

def process_czi_image(file_path, channels=None, slices=None, scale=1.0):
    """CZI image processing with optional arguments for selecting channels, slides and scale"""
    # Open the CZI file and get the data
    with CziFile(file_path) as czi:
        image_data = czi.asarray()
        if len(image_data.shape) == 8:  # If there are tiles
            image_data = image_data[:, :, 0, :, :, :, :, :]
        max_slice = image_data.shape[3]
        max_channel = image_data.shape[2]
    
    # If channels or slides are not specified, use all
    if channels is None:
        channels = list(range(max_channel))  # Use all channels
    if slices is None:
        slices = list(range(max_slice))  # Use all slides
    
    # Call a function to create a combined image from selected channels and slides
    combined_image = create_combined_image(image_data, channels, slices, scale, max_channel, max_slice)
    img = Image.fromarray(combined_image)
    
    return img

def process_tif_image(file_path):
    img = Image.open(file_path)
    
    img_stack = []        
    # Проходим по каждому слою в многослойном изображении
    for i in range(img.n_frames):
        img.seek(i)
        
        # Преобразуем слой в массив и нормализуем в 8-битный диапазон
        layer = np.array(img, dtype=np.uint16)
        layer_max = layer.max()
        
        # Проверка на максимальное значение, чтобы избежать деления на ноль
        if layer_max > 0:
            layer_normalized = (layer * (255.0 / layer_max)).astype(np.uint8)
        else:
            layer_normalized = layer.astype(np.uint8)  # Если max=0, просто переводим в uint8
            
        # Дублируем слой по трем каналам для RGB
        rgb_layer = np.stack([layer_normalized] * 3, axis=-1)
        img_stack.append(rgb_layer)
    
    # Объединяем все слои в одно RGB-изображение, используя максимумы по каждому каналу
    combined_array = np.max(np.stack(img_stack), axis=0)
    
    # Преобразуем в PIL-изображение и гарантируем режим RGB
    combined_img = Image.fromarray(combined_array, mode='RGB')
    if combined_img.mode != 'RGB':
        combined_img = combined_img.convert('RGB')
    
    return combined_img

def process_lif_image(file_path, scale=0.3):
    lif = LifFile(file_path)
    
    # Извлекаем, конвертируем в RGB, масштабируем и добавляем рамку к каждому изображению
    border_color = (128, 40, 128)  # цвет рамки
    border_size = 5  # Толщина рамки
    pil_images = []
    for img in lif.get_iter_image():
        plane = img.get_plane().convert("RGB")
        
        # Масштабируем изображение
        new_size = (int(plane.width * scale), int(plane.height * scale))
        scaled_image = plane.resize(new_size, Image.LANCZOS)
        
        # Добавляем зеленую рамку
        bordered_image = ImageOps.expand(scaled_image, border=border_size, fill=border_color)
        pil_images.append(bordered_image)

    # Определяем размеры для коллажа
    rows = int(len(pil_images) ** 0.5)
    cols = (len(pil_images) + rows - 1) // rows

    # Получаем размер каждого изображения с учетом рамки
    width, height = pil_images[0].size
    collage_width = cols * width
    collage_height = rows * height

    # Создаем пустой холст для коллажа
    collage = Image.new('RGB', (collage_width, collage_height))

    # Размещаем каждое изображение с рамкой в коллаже
    for i, img in enumerate(pil_images):
        x = (i % cols) * width
        y = (i // cols) * height
        collage.paste(img, (x, y))

    return collage

def process_synCatch_image(file_path):
    if file_path.endswith('.czi'):
        return process_czi_image(file_path)
    elif file_path.endswith('.tif'):
        return process_tif_image(file_path)
    elif file_path.endswith('.lif'):
        return process_lif_image(file_path)

# Function to invert an image
def invert_image(image):
    return 255 - image

def simplify_contour(coords, epsilon=1.0):
    """
    Simplify the ROI outline using the Douglas-Pecker algorithm to reduce the number of points.
    epsilon: Parameter to control the simplification level.
    """
    coords = np.array(coords, dtype=np.float32)
    approx_coords = cv2.approxPolyDP(coords, epsilon, True)  # True means the loop is closed
    return approx_coords.reshape(-1, 2) # Convert back to 2D array

def create_region_mask(image_shape, coords, simplify=True, epsilon=1.0):
    """
    Create a binary mask for a given set of coordinates.
    Optional simplification of the contour can be applied.
    """
    # Optionally simplify the coordinates to reduce the number of points
    if simplify:
        coords = simplify_contour(coords, epsilon)

    # Initialize a blank mask with the same height and width as the input image
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Single-channel mask (grayscale)

    # Fill the mask with the polygon defined by the simplified coordinates
    cv2.fillPoly(mask, [np.int32(coords)], 1)
    
    return mask


class ColorCycler:
    def __init__(self, num_colors=10):
        # Инициализируем палитру из N цветов
        self.palette = self.generate_color_palette(num_colors)
        self.index = 0  # Текущий индекс

    def generate_color_palette(self, num_colors):
        colormap = cm.get_cmap('tab10', num_colors)  # Используем палитру 'tab10'
        colors = [tuple((np.array(colormap(i)[:3]) * 255).astype(int)) for i in range(num_colors)]
        return colors

    def get_next_color(self):
        # Возвращаем цвет по кругу, используя индекс
        color = self.palette[self.index]
        self.index = (self.index + 1) % len(self.palette)  # Циклическое обновление индекса
        return color    
class guiButton:
    def __init__(self, x, y, width, height, text, callback=None):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.callback = callback  # Function that is called when the button is clicked
        self.visible = True

    def draw(self, img):
        if self.visible:
            cv2.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), (200, 200, 200), -1)
            cv2.putText(img, self.text, (self.x + 10, self.y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    def is_clicked(self, x, y):
        if self.visible and self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height:
            if self.callback:
                self.callback()
            return True
        return False



def draw_polygons_on_image(coords_df, scale_factor, color_cycler, img, simplify_contour):
    """
    Function to draw polygons on the image.

    Arguments:
    coords_df -- DataFrame with polygon coordinates
    scale_factor -- scaling factor for the coordinates
    color_cycler -- object to get the color of the polygon
    img -- image where polygons will be drawn
    simplify_contour -- function to simplify polygon coordinates
    """
    # If there are saved polygons, draw them
    if coords_df is not None:
        for col_x in coords_df.columns[::2]:  # Loop through every second column (assumed to be '_x' columns)
            col_y = col_x.replace('_x', '_y')  # Find the corresponding '_y' column
            location_name = col_x.rsplit('_', 1)[0]  # Extract the location name
            
            # Get the coordinates, apply the scaling factor
            coords = coords_df[[col_x, col_y]].values.astype(np.float32) * scale_factor
            # Simplify the coordinates using the provided function
            coords = simplify_contour(coords)
            # Convert the coordinates to integer type
            coords = coords.astype(np.int32)

            # Get the next color from the ColorCycler
            polygon_color = color_cycler.get_next_color()
            # Convert the color to integer tuple
            polygon_color = tuple(map(int, polygon_color))

            # Create a translucent layer for the polygon
            overlay = img.copy()
            # Fill the polygon with the chosen color
            cv2.fillPoly(overlay, [coords], color=polygon_color)
            # Blend the overlay with the original image to add transparency
            cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)

            # Calculate the center of the polygon
            center_x = int(np.mean(coords[:, 0]))
            center_y = int(np.mean(coords[:, 1]))
            text_position = (center_x, center_y)

            # Display the location name at the center of the polygon
            cv2.putText(img, location_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)

    return img



def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def is_on_edge(point, poly, tolerance=5):
    # Определение, попадает ли клик на ребро полигона
    for i in range(len(poly)):
        next_point = poly[(i + 1) % len(poly)]
        d = np.cross(next_point - poly[i], point - poly[i]) / distance(poly[i], next_point)
        if abs(d) < tolerance:
            # Если клик на ребре, возвращаем индекс ребра и ближайшую точку на ребре
            vec = next_point - poly[i]
            t = np.dot(point - poly[i], vec) / np.dot(vec, vec)
            if 0 <= t <= 1:
                nearest_point = poly[i] + t * vec
                return i, nearest_point.astype(np.int32)
    return None, None

class PolygonDrawer:
    def __init__(self, rgb_image, scale_factor=1.0, coords_df=None, comments=''):

        self.scale_factor = scale_factor
        self.original_rgb_image = rgb_image
        self.rgb_image = cv2.resize(rgb_image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        self.bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        self.img_copy = self.bgr_image.copy()
        self.comments = comments
         
        self.coords_df = coords_df  # координаты других полигонов

        self.points = []
        self.is_drawing = False
        self.tool_selected = False  # Устанавливается True после нажатия "Start"
        self.selected_vertex = None  # Индекс выбранной вершины
        self.dragging = False  # Флаг перетаскивания вершины
        self.tolerance = 10  # Радиус для определения клика на вершину

        # Создаем кнопки
        self.start_button = guiButton(10, 10, 100, 50, 'Start', self.start_drawing)
        self.delete_button = guiButton(10, 70, 100, 50, 'Delete', self.delete_polygon)
        self.select_all_button = guiButton(10, 130, 100, 50, 'Select all', self.select_all)
        self.apply_button = guiButton(10, 190, 100, 50, 'Apply', self.apply_polygon)
        self.cancel_button = guiButton(10, 250, 100, 50, 'Cancel', self.cancel_polygon)
        self.modify_button = guiButton(10, 310, 100, 50, 'Modify', self.modify_selected_polygon)  # Новая кнопка

        # Управляем видимостью кнопок
        self.cancel_button.visible = True
        self.apply_button.visible = False
        self.delete_button.visible = False
        self.modify_button.visible = False  # Скрыта по умолчанию

        cv2.namedWindow("Polygon")
        cv2.setMouseCallback("Polygon", self.mouse_callback)  # Обработчик мыши для рисования полигона

    def start_drawing(self):
        """ Метод для начала рисования """
        self.tool_selected = True
        self.start_button.visible = False  # Скрываем кнопку "Start"
        self.select_all_button.visible = False  # Скрываем кнопку "Select all"
        self.cancel_button.visible = False  # Скрываем кнопку "Cancel"
        cv2.setMouseCallback("Polygon", self.mouse_callback)  # Включаем режим рисования полигона
        # print("Draw tool selected")

    def apply_polygon(self):
        original_points = [(int(x / self.scale_factor), int(y / self.scale_factor)) for x, y in self.points]
        cv2.destroyAllWindows()  # Закрываем окно
        return original_points

    def modify_selected_polygon(self):
        """ Упрощает и активирует режим модификации полигона """
        self.points = simplify_contour(self.points, epsilon=1.0)  # Упрощаем координаты полигона
        self.modify_button.visible = False  # Скрываем кнопку после начала модификации
        cv2.setMouseCallback("Polygon", self.mod_mouse_callback)  # Включаем режим модификации полигона

    def delete_polygon(self):
        self.points = []  # Очищаем полигон
        self.tool_selected = False  # Сбрасываем флаг
        self.start_button.visible = True  # Показываем кнопку "Start"
        self.select_all_button.visible = True  # Показываем кнопку "Select all"
        self.delete_button.visible = False  # Скрываем кнопку "Delete"
        self.apply_button.visible = False  # Скрываем кнопку "Apply"
        self.modify_button.visible = False  # Скрываем кнопку "Modify Selected"

    def select_all(self):
        """ Выбирает все углы изображения для создания полигона """
        height, width = self.bgr_image.shape[:2]
        self.points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        self.delete_button.visible = True  # Показываем кнопку "Delete" после выбора всех углов
        self.select_all_button.visible = False  # Скрываем кнопку "Select all"
        self.start_button.visible = False  # Скрываем кнопку "Start"
        self.apply_button.visible = True  # Показываем кнопку "Apply"

    def cancel_polygon(self):
        """ Отменяет рисование полигона """
        self.points = []  # Очищаем массив координат
        cv2.destroyAllWindows()  # Закрываем окно
        return self.points  # Возвращаем пустой список

    def mouse_callback(self, event, x, y, flags, param):
        """ Обработчик мыши для рисования нового полигона """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Проверка нажатия на кнопки
            if self.start_button.is_clicked(x, y):
                return
            if self.delete_button.is_clicked(x, y):
                return
            if self.select_all_button.is_clicked(x, y):
                return
            if self.apply_button.is_clicked(x, y):
                return
            if self.cancel_button.is_clicked(x, y):
                return
            if self.modify_button.is_clicked(x, y):
                return

            if self.tool_selected:
                self.is_drawing = True
                self.points = [(x, y)]  # Начинаем новый полигон с первой точки

        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            if self.is_drawing:
                self.points.append((x, y))  # Добавляем точки в процессе рисования

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                if len(self.points) > 1:
                    self.delete_button.visible = True  # Показываем кнопку "Delete" после завершения рисования
                    self.apply_button.visible = True  # Показываем кнопку "Apply"
                    self.modify_button.visible = True  # Показываем кнопку "Modify Selected"

    def mod_mouse_callback(self, event, x, y, flags, param):
        """ Обработчик мыши для модификации полигона """
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Проверка нажатия на кнопки
            if self.start_button.is_clicked(x, y):
                return
            if self.delete_button.is_clicked(x, y):
                return
            if self.select_all_button.is_clicked(x, y):
                return
            if self.apply_button.is_clicked(x, y):
                return
            if self.cancel_button.is_clicked(x, y):
                return
            if self.modify_button.is_clicked(x, y):
                return

            # Проверка клика на вершину полигона для модификации
            for i, point in enumerate(self.points):
                if distance(np.array(point), np.array([x, y])) < self.tolerance:
                    self.selected_vertex = i
                    self.dragging = True
                    return

            # Проверка на ребро полигона
            edge_index, new_vertex = is_on_edge(np.array([x, y]), np.array(self.points))
            if edge_index is not None:
                next_index = (edge_index + 1) % len(self.points)
                
                # Преобразуем self.points в список для работы с insert
                self.points = self.points.tolist()
                # Вставляем новую точку
                self.points.insert(next_index, tuple(new_vertex))
                # Преобразуем self.points обратно в numpy array
                self.points = np.array(self.points)

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            # Удаление вершины при двойном клике
            for i, point in enumerate(self.points):
                if distance(np.array(point), np.array([x, y])) < self.tolerance:
                    # Удаляем вершину
                    self.points = self.points.tolist()
                    self.points.pop(i)
                    self.points = np.array(self.points)
                    return  # Завершаем выполнение после удаления

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_vertex is not None:
                self.points[self.selected_vertex] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.selected_vertex = None

    def run(self):
        while True:
            img = self.img_copy.copy()

            # Рисуем полигоны, если есть другие сохраненные полигоны
            color_cycler = ColorCycler(num_colors=10)
            img = draw_polygons_on_image(self.coords_df, self.scale_factor, color_cycler, img, simplify_contour)

            # Рисуем кнопки
            self.start_button.draw(img)
            self.delete_button.draw(img)
            self.select_all_button.draw(img)
            self.apply_button.draw(img)
            self.cancel_button.draw(img)
            self.modify_button.draw(img)

            # Рисуем полигон, который пользователь сейчас рисует
            if len(self.points) > 0:
                # Преобразуем точки в массив numpy для работы с cv2
                polygon_points = np.array(self.points, np.int32)
                
                # Рисуем замкнутый полигон (isClosed=True)
                cv2.polylines(img, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

                # Рисуем точки на вершинах
                for point in self.points:
                    # Преобразуем координаты точки в целые числа
                    cv2.circle(img, (int(point[0]), int(point[1])), radius=5, color=(0, 255, 0), thickness=-1)  # точки

            
            ## ОТОБРАЖЕНИЕ comments
            # Определяем параметры текста
            text = self.comments  # Используем переменную comments
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1

            # Получаем размеры текста и координаты для его центрирования
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            img_height, img_width = img.shape[:2]
            x = (img_width - text_width) // 2
            y = text_height + 20  # Отступ от верхнего края

            # Рисуем серый полупрозрачный фон
            overlay = img.copy()
            cv2.rectangle(overlay, (x - 10, y - text_height - 10), (x + text_width + 10, y + 10), (128, 128, 128), -1)
            alpha = 0.5  # Прозрачность фона (0 полностью прозрачно, 1 полностью непрозрачно)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Наносим белый текст поверх
            cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            # ОТОБРАЖЕНИЕ изображение img
            cv2.imshow("Polygon", img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or cv2.getWindowProperty("Polygon", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        original_points = [(int(x / self.scale_factor), int(y / self.scale_factor)) for x, y in self.points]
        return original_points



class PolygonModifier:
    def __init__(self, rgb_image, scale_factor=1.0, coords_df=None):
        self.scale_factor = scale_factor
        self.rgb_image = cv2.resize(rgb_image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        self.bgr_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_RGB2BGR)
        self.img_copy = self.bgr_image.copy()

        self.coords_df = coords_df  # DataFrame с координатами полигонов
        self.selected_polygon_name = None
        self.selected_polygon_points = None

        # Создание кнопок
        self.start_button = guiButton(10, 10, 100, 50, 'Start', self.start_selection)
        self.delete_button = guiButton(10, 70, 100, 50, 'Delete', self.delete_polygon)
        self.modify_button = guiButton(10, 130, 100, 50, 'Modify', self.modify_polygon)
        self.cancel_button = guiButton(10, 190, 100, 50, 'Cancel', self.cancel_modification)

        # Изначально кнопки "Delete" и "Modify" скрыты
        self.delete_button.visible = False
        self.modify_button.visible = False

        cv2.namedWindow("Polygon")
        cv2.setMouseCallback("Polygon", self.handle_mouse)

    def start_selection(self):
        """Активируем инструмент выбора полигонов"""
        self.start_button.visible = False

    def handle_mouse(self, event, x, y, flags, param):
        """Обработка нажатий мыши для выбора полигона"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Проверяем кнопки
            if self.start_button.is_clicked(x, y):
                return
            if self.delete_button.is_clicked(x, y):
                return
            if self.modify_button.is_clicked(x, y):
                return
            if self.cancel_button.is_clicked(x, y):
                return

            # Определение, находится ли клик внутри полигона
            for col_x in self.coords_df.columns[::2]:
                col_y = col_x.replace('_x', '_y')
                polygon_points = self.coords_df[[col_x, col_y]].values.astype(np.float32) * self.scale_factor

                if cv2.pointPolygonTest(np.array(polygon_points, np.int32), (x, y), False) >= 0:
                    self.selected_polygon_name = col_x.rsplit('_', 1)[0]
                    self.selected_polygon_points = polygon_points
                    self.delete_button.visible = True
                    self.modify_button.visible = True
                    break

    def delete_polygon(self):
        """Удаление выбранного полигона"""
        if self.selected_polygon_name:
            col_x = f'{self.selected_polygon_name}_x'
            col_y = f'{self.selected_polygon_name}_y'
            # Удаляем полигон из DataFrame
            self.coords_df.drop([col_x, col_y], axis=1, inplace=True)
            cv2.destroyWindow("Polygon")  # Закрываем окно

    def modify_polygon(self):
        """Просто закрываем окно для дальнейшей обработки координат"""
        cv2.destroyWindow("Polygon")  # Закрываем окно

    def cancel_modification(self):
        """Закрытие окна без изменений"""
        self.selected_polygon_name = None
        self.selected_polygon_points = None
        cv2.destroyWindow("Polygon")  # Закрываем окно

    def run(self):
        """Основной цикл отрисовки интерфейса"""
        while True:
            img = self.img_copy.copy()

            # Отрисовка всех полигонов
            for col_x in self.coords_df.columns[::2]:
                col_y = col_x.replace('_x', '_y')
                polygon_points = self.coords_df[[col_x, col_y]].values.astype(np.float32) * self.scale_factor
                color = (0, 255, 0) if col_x == f'{self.selected_polygon_name}_x' else (255, 0, 0)
                cv2.polylines(img, [np.array(polygon_points, np.int32)], isClosed=True, color=color, thickness=2)

                # Подпись имени региона в центре полигона, обработка NaN
                center_x = int(np.nanmean(polygon_points[:, 0]))
                center_y = int(np.nanmean(polygon_points[:, 1]))
                text_position = (center_x, center_y)
                location_name = col_x.rsplit('_', 1)[0]
                cv2.putText(img, location_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Отображение кнопок
            self.start_button.draw(img)
            self.delete_button.draw(img)
            self.modify_button.draw(img)
            self.cancel_button.draw(img)

            cv2.imshow("Polygon", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or cv2.getWindowProperty("Polygon", cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()

# # Example of using the class
# image_path = r"E:\iMAGES\4 months slide2 slice5\4 months slide2 slice5_Experiment-1271_synaptotag.png"
# rgb_image = cv2.imread(image_path)  # Load the image into BGR
# rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

# drawer = PolygonDrawer(rgb_image)
# polygon_points = drawer.run()
# print("Coordinates of the polygon:", polygon_points)



class ParallelogramEditor:
    def __init__(self, image, scale_factor=1.0, coords_df=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.scale_factor = scale_factor
        self.coords_df = coords_df # coordinates of other polygons
        
        #self.original_image = image
        self.image = cv2.resize(image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
        
        # Инициализируем ColorCycler
        color_cycler = ColorCycler(num_colors=10)
        self.image = draw_polygons_on_image(self.coords_df, self.scale_factor, color_cycler, self.image, simplify_contour)
        
        self.original_image = self.image.copy()
        self.drawing = False
        self.start_point = None
        self.points = []
        self.rotated_points = []
        self.finished_drawing = False
        self.moving_point = None
        self.rotating = False
        self.center_point = None
        self.initial_angle = 0
        self.dragging = False
        
        # Create a window
        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', self.draw_parallelogram)

    def rotate_point(self, point, center, angle):
        """Rotates a point around the center by a specified angle."""
        angle_rad = math.radians(angle)
        x_new = int(center[0] + math.cos(angle_rad) * (point[0] - center[0]) - math.sin(angle_rad) * (point[1] - center[1]))
        y_new = int(center[1] + math.sin(angle_rad) * (point[0] - center[0]) + math.cos(angle_rad) * (point[1] - center[1]))
        return (x_new, y_new)

    def draw_polygon(self, img, pts):
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    def draw_grid(self, img, points, step_fraction=0.1):
        """Draws a mesh inside a parallelogram."""
        # Break the parallelogram into vectors
        vector_u = np.array(points[1]) - np.array(points[0])
        vector_v = np.array(points[3]) - np.array(points[0])
    
        # Determine the number of steps along each side
        num_steps_x = int(1 / step_fraction)
        num_steps_y = int(1 / step_fraction)
    
        # Draw vertical grid lines
        for i in range(1, num_steps_x):
            step_u = vector_u * (i * step_fraction)
            start_point = np.array(points[0]) + step_u
            end_point = np.array(points[3]) + step_u
            cv2.line(img, tuple(start_point.astype(int)), tuple(end_point.astype(int)), (255, 0, 0), 1)
    
        # Draw horizontal grid lines
        for j in range(1, num_steps_y):
            step_v = vector_v * (j * step_fraction)
            start_point = np.array(points[0]) + step_v
            end_point = np.array(points[1]) + step_v
            cv2.line(img, tuple(start_point.astype(int)), tuple(end_point.astype(int)), (255, 0, 0), 1)


    def draw_parallelogram(self, event, x, y, flags, param):
        if not self.finished_drawing:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
                self.points = [self.start_point]
                #print('Drawing started')
    
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.image = self.original_image.copy()
                end_point = (x, y)
                cv2.rectangle(self.image, self.start_point, end_point, (0, 255, 0), 2)
    
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                end_point = (x, y)
                self.points = [self.start_point, (end_point[0], self.start_point[1]), end_point, (self.start_point[0], end_point[1])]
    
                self.finished_drawing = True
                self.center_point = ((self.points[0][0] + self.points[2][0]) // 2, (self.points[0][1] + self.points[2][1]) // 2)
                #print('Drawing stopped')
    
                # Draw a figure
                self.draw_polygon(self.image, self.points)
                self.draw_grid(self.image, self.points)  # Add a grid
    
        else:
            if event == cv2.EVENT_RBUTTONDOWN:
                self.rotating = True
                self.initial_angle = math.degrees(math.atan2(y - self.center_point[1], x - self.center_point[0]))
                #print('Rotation started')
    
            elif event == cv2.EVENT_RBUTTONUP:
                if self.rotating:
                    #print('Rotation stopped')
                    self.rotating = False
                    self.points = self.rotated_points
    
            elif event == cv2.EVENT_LBUTTONDOWN:
                # Determine which vertex is moving or start moving the entire figure
                for i, point in enumerate(self.points):
                    if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                        self.moving_point = i
                        #print(f'Started moving point {i}')
                        break
                else:
                    # Check if the click is inside the shape to move the entire shape
                    if self.is_point_inside_polygon((x, y), self.points):
                        self.dragging = True
                        self.drag_start = (x, y)
                        self.original_points = self.points.copy()
                        #print('Started dragging the whole figure')
    
            elif event == cv2.EVENT_MOUSEMOVE:
                if self.moving_point is not None:
                    self.image = self.original_image.copy()
    
                    # Move selected vertex
                    self.points[self.moving_point] = (x, y)
    
                    # Automatic recalculation of point 2
                    if self.moving_point in [0, 1, 3]:
                        vector_01 = np.array(self.points[1]) - np.array(self.points[0])
                        vector_03 = np.array(self.points[3]) - np.array(self.points[0])
                        self.points[2] = tuple(np.array(self.points[0]) + vector_01 + vector_03)
    
                    # Draw only lines 0-1 and 0-3
                    cv2.line(self.image, tuple(self.points[0]), tuple(self.points[1]), (0, 255, 255), 2)  # Yellow color
                    cv2.line(self.image, tuple(self.points[0]), tuple(self.points[3]), (0, 255, 255), 2)  # Yellow color
                    self.draw_grid(self.image, self.points)  # Add a grid
    
                elif self.dragging:
                    # Move the entire shape
                    dx = x - self.drag_start[0]
                    dy = y - self.drag_start[1]
                    self.points = [(px + dx, py + dy) for px, py in self.original_points]
    
                    self.image = self.original_image.copy()
                    self.draw_polygon(self.image, self.points)
                    self.draw_grid(self.image, self.points)  # Add a grid
    
                elif self.rotating:
                    # Calculate the rotation angle
                    current_angle = math.degrees(math.atan2(y - self.center_point[1], x - self.center_point[0]))
                    angle_diff = current_angle - self.initial_angle
                    self.rotated_points = [self.rotate_point(point, self.center_point, angle_diff) for point in self.points]
    
                    self.image = self.original_image.copy()
                    self.draw_polygon(self.image, self.rotated_points)
                    self.draw_grid(self.image, self.rotated_points)  # Add a grid
    
            elif event == cv2.EVENT_LBUTTONUP:
                if self.moving_point is not None:
                    #print(f'Stopped moving point {self.moving_point}')
                    self.moving_point = None
    
                if self.dragging:
                    #print('Stopped dragging the whole figure')
                    self.dragging = False
    
        if keyboard.is_pressed('ctrl') and self.finished_drawing:
            self.image = self.original_image.copy()
            # Draw only lines 0-1 and 0-3
            cv2.line(self.image, tuple(self.points[0]), tuple(self.points[1]), (0, 255, 255), 2)  # Yellow color
            cv2.line(self.image, tuple(self.points[0]), tuple(self.points[3]), (0, 255, 255), 2)  # Yellow color
            # Draw the mesh based on the updated points
            self.draw_grid(self.image, self.points)  # The grid is also in yellow mode
    
    def is_point_inside_polygon(self, point, polygon):
        """Checks if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    

    def get_coordinates(self):
        """Returns the current coordinates of the parallelogram, taking into account scaling."""
        if not self.finished_drawing:
            return None
        
        current_coordinates = self.rotated_points if self.rotating else self.points
        # Rescaling coordinates to the original image size
        scaled_coordinates = [(int(x / self.scale_factor), int(y / self.scale_factor)) for x, y in current_coordinates]
        
        return scaled_coordinates


    def run(self):
        while True:
            cv2.imshow('Image', self.image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Press 'q' to exit
                break
            # Check for closing the window manually
            if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()

# # Example of using the class
# if __name__ == "__main__":
#     image_path = r"E:\iMAGES\P21.5.1 slide2 slice4\P21.5.1 slide2 slice4_Experiment-888_comments.png"
#     image = cv2.imread(image_path)
    
#     # Initialize the class with an image
#     editor = ParallelogramEditor(image)
#     editor.run()
#     coords = editor.get_coordinates()
#     print("Final coordinates of the parallelogram:", coords)

def read_metadata_from_file(file_path):
    ads_path = f"{file_path}:syn_catch_metadata"
    if os.path.exists(ads_path):
        with open(ads_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def metadata_to_text(metadata, exclude_keys=None, comments=None):
    exclude_keys = exclude_keys or []
    comments = comments or {}
    # Исключаем ненужные ключи
    filtered_metadata = {key: value for key, value in metadata.items() if key not in exclude_keys}
    # Формируем текст из метаданных
    lines = []
    for key, value in filtered_metadata.items():
        line = f"{key}: {value}"
        if key in comments:
            line += f"  # {comments[key]}"
        lines.append(line)
    return "\n".join(lines)
            
class ExperimentWindow:
    def __init__(self, experiment_path, comment=None):
        self.experiment_path = experiment_path
        self.current_image = None
        self.current_image_path = None
        self.first_image_displayed = False
        extension = os.path.splitext(experiment_path)[1]
        self.results_folder = self.experiment_path.replace(extension, "_results")
        self.multiple_pictures_mode = False
        self.next_canvas = 1
        self.current_images = [None, None]  # Для режима множественного выбора
        self.current_image_paths = [None, None]

        # Создание дополнительного окна
        self.top = Toplevel()
        self.top.title("Gallery of Experiment")
        self.top.geometry("1024x768")
        self.top.iconbitmap(icon_path)  # Установка иконки окна

        
        # Фрейм для миниатюр
        self.thumbnail_frame = Frame(self.top)

        # Разворачиваем окно на весь экран
        self.top.wm_state("zoomed")

        # Метка для отображения имени файла
        self.file_name_label = Label(self.top, text="", font=("Arial", 14))

        self.toolbar_frame = Frame(self.top, height=15)

        # Фрейм для изображения
        self.large_image_frame = Frame(self.top)

        # Перемещаем слайдер масштаба в toolbar_frame
        self.scale_slider = Scale(
            self.toolbar_frame,
            from_=10,
            to=200,
            orient=tk.HORIZONTAL,
            command=self.update_image
        )
        self.scale_slider.set(100)

        # Canvas для большого изображения
        self.large_image_canvas = Canvas(self.large_image_frame, width=500, height=400, highlightthickness=0)

        # Скроллбары для canvas
        self.y_scrollbar = tk.Scrollbar(
            self.large_image_frame, orient=tk.VERTICAL, command=self.large_image_canvas.yview
        )
        self.x_scrollbar = tk.Scrollbar(
            self.large_image_frame, orient=tk.HORIZONTAL, command=self.large_image_canvas.xview
        )

        # Привязываем обработчики кликов на canvas
        self.large_image_canvas.bind("<Double-Button-1>", self.on_double_click)
        self.large_image_canvas.bind("<Button-1>", self.on_single_click)

        self.open_folder_button = Button(
            self.toolbar_frame, text="Open Results Folder", command=self.open_results_folder
        )
        self.remove_results_button = Button(
            self.toolbar_frame, text="Clear Results", command=self.remove_results
        )

        # Добавляем кнопку "Review multiple pictures"
        self.select_multiple_button = Checkbutton(
            self.toolbar_frame, text="Review multiple pictures", command=self.toggle_multiple_pictures_mode,
            variable=tk.BooleanVar(value=False)
        )
        
        # Отображение метаданных
        #self.metadata_label = Label(self.toolbar_frame, text="", font=("Arial", 10), justify=tk.LEFT, anchor="w")
        self.metadata_text, self.metadata_scrollbar = create_metadata_text_field(self.toolbar_frame)

        # Расположение элементов
        self.toolbar_frame.pack(fill=tk.BOTH, expand=False)
        self.open_folder_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.remove_results_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.select_multiple_button.pack(side=tk.LEFT, padx=5, pady=5)
        #self.metadata_label.pack(side=tk.LEFT, padx=10)
        self.metadata_text.pack(side=tk.LEFT, padx=10, pady=5)
        self.metadata_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        self.scale_slider.pack(side=tk.RIGHT, padx=5, pady=5)
        self.file_name_label.pack(pady=5)
        self.large_image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.large_image_canvas.config(
            xscrollcommand=self.x_scrollbar.set, yscrollcommand=self.y_scrollbar.set
        )
        self.large_image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.thumbnail_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        self.top.focus_force()
        
        # Привязываем колесико мышки к большому canvas
        self.large_image_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.y_scrollbar.bind(
            "<MouseWheel>",
            lambda event: self.on_mouse_wheel_scroll(event, self.large_image_canvas, 'y')
        )
        self.x_scrollbar.bind(
            "<MouseWheel>",
            lambda event: self.on_mouse_wheel_scroll(event, self.large_image_canvas, 'x')
        )

        # Если комментарий передан, отображаем его
        if comment:
            comment_label = Label(self.top, text=comment, font=("Arial", 12), fg="blue")
            comment_label.pack(pady=5)

        # Получаем все изображения результата
        images, metadatas = self.get_results_data(self.experiment_path)
         
        # Используем функцию display_thumbnails
        self.display_thumbnails_in_class(images, metadatas)

        # Привязываем событие закрытия окна
        self.top.protocol("WM_DELETE_WINDOW", self.on_close)

    # Функция для получения всех результатов в папке по пути к эксперименту
    def get_results_data(self, experiment_path):
        extension = os.path.splitext(experiment_path)[1]
        result_folder = experiment_path.replace(extension, "_results")
        if os.path.exists(result_folder):
            images = [
                os.path.join(result_folder, f)
                for f in os.listdir(result_folder)
                if f.endswith('.png')
            ]
            metadata_texts = []
            for image_path in images:
                metadata = read_metadata_from_file(image_path)
                metadata_texts.append(metadata_to_text(metadata, exclude_keys = ['protocol']))
            return images, metadata_texts
        return [], []

    def display_thumbnails_in_class(self, images, metadatas):
        # sort data by date
        image_metadata_pairs = zip(images, metadatas)
        sorted_pairs = sorted(image_metadata_pairs, key=lambda x: os.path.getctime(x[0]))
        sorted_images, sorted_metadatas = zip(*sorted_pairs)
        
        # Определяем функции обработки событий
        def on_single_click(path):
            # При клике отображаем большое изображение
            self.display_large_image(path)
            metadata = read_metadata_from_file(path)
            self.display_metadata(metadata)

        def on_double_click(path):            
            pass  # Можно добавить логику по необходимости

        # Уничтожаем предыдущий экземпляр ThumbnailViewer, если он существует
        if hasattr(self, 'thumbnail_viewer'):
            self.thumbnail_viewer.destroy()

        # Создаем новый экземпляр ThumbnailViewer
        self.thumbnail_viewer = ThumbnailViewer(
            parent=self.thumbnail_frame,
            images=sorted_images,
            replaced_image_names = sorted_metadatas,
            comments=None,  # Можно передать список комментариев
            on_single_click=on_single_click,
            on_double_click=on_double_click,
            open_image_func=Image.open,
            max_per_page=None,  # Показать все изображения
            width=500,
            height=150,
        )

        # Устанавливаем фокус на thumbnail_inner_frame
        self.thumbnail_viewer.thumbnail_inner_frame.focus_set()


        # Применяем тему через ThemeManager
        theme_manager = ThemeManager()
        theme_manager.apply_theme(self.top)
        
        
    # Функция для переключения режима множественного выбора
    def toggle_multiple_pictures_mode(self):
        self.multiple_pictures_mode = not self.multiple_pictures_mode
        if self.multiple_pictures_mode:
            
            # Очищаем окно метаданных
            self.display_metadata({})
            
            # Изменяем вид кнопки на нажатую
            #self.select_multiple_button.config(relief=tk.SUNKEN)
            # Убираем элементы одноканального режима
            self.large_image_canvas.pack_forget()
            self.scale_slider.set(100)  # Сбрасываем масштаб
            self.y_scrollbar.pack_forget()
            self.x_scrollbar.pack_forget()

            # Создаем два canvas и размещаем их
            self.large_image_canvas1_frame = Frame(self.large_image_frame)
            self.large_image_canvas2_frame = Frame(self.large_image_frame)
            

            # Канвасы
            self.large_image_canvas1 = Canvas(
                self.large_image_canvas1_frame, width=500, height=400, highlightthickness=0
            )
            self.large_image_canvas2 = Canvas(
                self.large_image_canvas2_frame, width=500, height=400, highlightthickness=0
            )

            # Скроллбары для канвасов
            self.y_scrollbar1 = tk.Scrollbar(
                self.large_image_canvas1_frame, orient=tk.VERTICAL, command=self.sync_scroll1
            )
            self.y_scrollbar2 = tk.Scrollbar(
                self.large_image_canvas2_frame, orient=tk.VERTICAL, command=self.sync_scroll2
            )

            self.x_scrollbar1 = tk.Scrollbar(
                self.large_image_canvas1_frame, orient=tk.HORIZONTAL, command=self.sync_scroll_x1
            )
            self.x_scrollbar2 = tk.Scrollbar(
                self.large_image_canvas2_frame, orient=tk.HORIZONTAL, command=self.sync_scroll_x2
            )

            # Настраиваем канвасы
            self.large_image_canvas1.config(
                yscrollcommand=self.y_scrollbar1.set, xscrollcommand=self.x_scrollbar1.set
            )
            self.large_image_canvas2.config(
                yscrollcommand=self.y_scrollbar2.set, xscrollcommand=self.x_scrollbar2.set
            )

            # Размещаем элементы в первом фрейме с использованием grid
            self.large_image_canvas1.grid(row=0, column=0, sticky="nsew")
            self.y_scrollbar1.grid(row=0, column=1, sticky="ns")
            self.x_scrollbar1.grid(row=1, column=0, sticky="ew")

            self.large_image_canvas1_frame.grid_rowconfigure(0, weight=1)
            self.large_image_canvas1_frame.grid_columnconfigure(0, weight=1)

            # Размещаем элементы во втором фрейме с использованием grid
            self.large_image_canvas2.grid(row=0, column=0, sticky="nsew")
            self.y_scrollbar2.grid(row=0, column=1, sticky="ns")
            self.x_scrollbar2.grid(row=1, column=0, sticky="ew")

            self.large_image_canvas2_frame.grid_rowconfigure(0, weight=1)
            self.large_image_canvas2_frame.grid_columnconfigure(0, weight=1)

            # Размещаем фреймы канвасов
            self.large_image_canvas1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.large_image_canvas2_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            # Привязываем обработчики двойного клика к канвасам
            self.large_image_canvas1.bind("<Double-Button-1>", self.on_double_click_canvas1)
            self.large_image_canvas2.bind("<Double-Button-1>", self.on_double_click_canvas2)
            
            # Привязываем события прокрутки
            self.large_image_canvas1.bind("<MouseWheel>", self.on_mouse_wheel_canvas1)
            self.large_image_canvas2.bind("<MouseWheel>", self.on_mouse_wheel_canvas2)

            # Привязываем обработчики однократного клика к канвасам
            self.large_image_canvas1.bind("<Button-1>", self.on_single_click_canvas1)
            self.large_image_canvas2.bind("<Button-1>", self.on_single_click_canvas2)

            # Сбрасываем переключатель канвасов
            self.next_canvas = 1
        else:
            # Очищаем окно метаданных
            self.display_metadata({})
            # Изменяем вид кнопки на отжатую
            #self.select_multiple_button.config(relief=tk.RAISED)
            # Убираем канвасы множественного выбора
            self.large_image_canvas1_frame.pack_forget()
            self.large_image_canvas2_frame.pack_forget()
            # Сбрасываем текущие изображения
            self.current_images = [None, None]
            self.current_image_paths = [None, None]
            # Возвращаем элементы одноканального режима
            self.y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.large_image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.update_image(int(self.scale_slider.get()))  # Обновляем изображение

    # Функция для отображения большого изображения
    def display_large_image(self, image_path):
        self.current_image_path = image_path
        self.current_image = Image.open(image_path)
        file_name = os.path.basename(image_path)
        self.file_name_label.config(text=file_name)

        if not self.first_image_displayed:
            if not self.multiple_pictures_mode:
                # Рассчитываем начальный масштаб для одноканального режима
                canvas_height = self.large_image_canvas.winfo_height()
                img_width, img_height = self.current_image.size
                height_scale = canvas_height / img_height
                initial_scale = height_scale * 100
                self.scale_slider.set(int(initial_scale))
            self.first_image_displayed = True

        if self.multiple_pictures_mode:
            # В режиме множественного выбора отображаем изображения поочередно на двух канвасах
            if self.next_canvas == 1:
                self.current_images[0] = self.current_image
                self.current_image_paths[0] = self.current_image_path
                self.display_image_in_canvas(
                    self.current_image, self.large_image_canvas1, scale_value=int(self.scale_slider.get())
                )
                self.next_canvas = 2
            else:
                self.current_images[1] = self.current_image
                self.current_image_paths[1] = self.current_image_path
                self.display_image_in_canvas(
                    self.current_image, self.large_image_canvas2, scale_value=int(self.scale_slider.get())
                )
                self.next_canvas = 1
            # Синхронизируем scrollregion после загрузки изображений
            self.sync_scroll_regions()
        else:
            self.update_image(int(self.scale_slider.get()))

    # Функция для отображения изображения на заданном канвасе
    def display_image_in_canvas(self, image, canvas, scale_value=100):
        scale_factor = int(float(scale_value)) / 100.0
        new_size = (
            int(image.size[0] * scale_factor),
            int(image.size[1] * scale_factor)
        )
        resized_image = image.resize(new_size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor='nw', image=img_tk)
        canvas.image = img_tk
        canvas.config(scrollregion=canvas.bbox("all"))

    # Функция для изменения изображения при изменении масштаба
    def update_image(self, scale_value):
        if self.multiple_pictures_mode:
            # Обновляем изображения на обоих канвасах
            scale_factor = int(float(scale_value)) / 100.0

            # Проверка на наличие атрибутов large_image_canvas1 и large_image_canvas2
            if hasattr(self, 'large_image_canvas1') and hasattr(self, 'large_image_canvas2'):
                for i, canvas in enumerate([self.large_image_canvas1, self.large_image_canvas2]):
                    if self.current_images[i]:
                        resized_image = self.current_images[i].resize(
                            (
                                int(self.current_images[i].size[0] * scale_factor),
                                int(self.current_images[i].size[1] * scale_factor)
                            ),
                            Image.LANCZOS
                        )
                        img_tk = ImageTk.PhotoImage(resized_image)
                        canvas.delete("all")
                        canvas.create_image(0, 0, anchor='nw', image=img_tk)
                        canvas.image = img_tk
                        canvas.config(scrollregion=canvas.bbox("all"))
                # Синхронизируем scrollregion после изменения масштаба
                self.sync_scroll_regions()
            else:
                None
        else:
            if self.current_image:
                scale_factor = int(float(scale_value)) / 100.0
                new_size = (
                    int(self.current_image.size[0] * scale_factor),
                    int(self.current_image.size[1] * scale_factor)
                )
                resized_image = self.current_image.resize(new_size, Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(resized_image)

                # Очищаем canvas и добавляем изображение
                if hasattr(self, 'large_image_canvas'):
                    self.large_image_canvas.delete("all")
                    self.large_image_canvas.create_image(0, 0, anchor='nw', image=img_tk)
                    self.large_image_canvas.image = img_tk
                    self.large_image_canvas.config(scrollregion=self.large_image_canvas.bbox("all"))
                else:
                    None


    # Обработчик события колесика мыши
    def on_mouse_wheel(self, event):
        if event.state & 0x0001:  # Если Shift нажат
            if event.delta > 0:
                self.large_image_canvas.xview_scroll(-1, "units")
            else:
                self.large_image_canvas.xview_scroll(1, "units")
        else:
            if event.delta > 0:
                self.large_image_canvas.yview_scroll(-1, "units")
            else:
                self.large_image_canvas.yview_scroll(1, "units")

    # Обработчики колесика мыши для канвасов в режиме множественного выбора
    def on_mouse_wheel_canvas1(self, event):
        if event.state & 0x0001:  # Если Shift нажат
            self.large_image_canvas1.xview_scroll(int(-1 * (event.delta / 120)), "units")
            self.sync_scroll_positions('canvas1', 'x')
        else:
            self.large_image_canvas1.yview_scroll(int(-1 * (event.delta / 120)), "units")
            self.sync_scroll_positions('canvas1', 'y')

    def on_mouse_wheel_canvas2(self, event):
        if event.state & 0x0001:  # Если Shift нажат
            self.large_image_canvas2.xview_scroll(int(-1 * (event.delta / 120)), "units")
            self.sync_scroll_positions('canvas2', 'x')
        else:
            self.large_image_canvas2.yview_scroll(int(-1 * (event.delta / 120)), "units")
            self.sync_scroll_positions('canvas2', 'y')

    # Функции для синхронизации прокрутки
    def sync_scroll1(self, *args):
        self.large_image_canvas1.yview(*args)
        self.large_image_canvas2.yview_moveto(self.large_image_canvas1.yview()[0])
        self.y_scrollbar2.set(*self.large_image_canvas1.yview())

    def sync_scroll2(self, *args):
        self.large_image_canvas2.yview(*args)
        self.large_image_canvas1.yview_moveto(self.large_image_canvas2.yview()[0])
        self.y_scrollbar1.set(*self.large_image_canvas2.yview())

    def sync_scroll_x1(self, *args):
        self.large_image_canvas1.xview(*args)
        self.large_image_canvas2.xview_moveto(self.large_image_canvas1.xview()[0])
        self.x_scrollbar2.set(*self.large_image_canvas1.xview())

    def sync_scroll_x2(self, *args):
        self.large_image_canvas2.xview(*args)
        self.large_image_canvas1.xview_moveto(self.large_image_canvas2.xview()[0])
        self.x_scrollbar1.set(*self.large_image_canvas2.xview())

    def sync_scroll_positions(self, source_canvas, direction):
        if source_canvas == 'canvas1':
            if direction == 'y':
                pos = self.large_image_canvas1.yview()
                self.large_image_canvas2.yview_moveto(pos[0])
                self.y_scrollbar2.set(*pos)
            else:
                pos = self.large_image_canvas1.xview()
                self.large_image_canvas2.xview_moveto(pos[0])
                self.x_scrollbar2.set(*pos)
        else:
            if direction == 'y':
                pos = self.large_image_canvas2.yview()
                self.large_image_canvas1.yview_moveto(pos[0])
                self.y_scrollbar1.set(*pos)
            else:
                pos = self.large_image_canvas2.xview()
                self.large_image_canvas1.xview_moveto(pos[0])
                self.x_scrollbar1.set(*pos)

    def sync_scroll_regions(self):
        # Synchronize the scrollregion size
        region1 = self.large_image_canvas1.bbox("all")
        region2 = self.large_image_canvas2.bbox("all")
        
        # Check for None regions and handle accordingly
        if region1 is None and region2 is None:
            # Both canvases are empty; nothing to synchronize
            return
        elif region1 is None:
            # Canvas1 is empty; use region2 dimensions
            max_width = region2[2]
            max_height = region2[3]
        elif region2 is None:
            # Canvas2 is empty; use region1 dimensions
            max_width = region1[2]
            max_height = region1[3]
        else:
            # Both canvases have items; take the maximum dimensions
            max_width = max(region1[2], region2[2])
            max_height = max(region1[3], region2[3])

        # Update scrollregion for both canvases
        self.large_image_canvas1.config(scrollregion=(0, 0, max_width, max_height))
        self.large_image_canvas2.config(scrollregion=(0, 0, max_width, max_height))


    def on_mouse_wheel_scroll(self, event, canvas, scroll_direction):
        if event.delta > 0:
            canvas.yview_scroll(-1, "units") if scroll_direction == 'y' else canvas.xview_scroll(-1, "units")
        else:
            canvas.yview_scroll(1, "units") if scroll_direction == 'y' else canvas.xview_scroll(1, "units")

    # Обработчик двойного клика для открытия изображения в проводнике
    def on_double_click(self, event):
        if self.current_image_path:
            self.open_file(self.current_image_path)

    def on_double_click_canvas1(self, event):
        if self.current_image_paths[0]:
            self.open_file(self.current_image_paths[0])

    def on_double_click_canvas2(self, event):
        if self.current_image_paths[1]:
            self.open_file(self.current_image_paths[1])
    
    # Функция для открытия файла в проводнике
    def open_file(self, file_path):
        file_path = file_path.replace('/', '\\')
        if os.path.exists(file_path):
            try:
                cmd = f'explorer "{file_path}"'
                subprocess.Popen(cmd)
            except Exception as e:
                print(f"Failed to open file '{file_path}': {e}")
        else:
            print(f"File '{file_path}' not found")

    def process_image_click(self, index):
        if self.current_image_paths[index]:
            image_path = self.current_image_paths[index]
            metadata = read_metadata_from_file(image_path)
            self.display_metadata(metadata)
            file_name = self.extract_file_name(image_path)
            self.file_name_label.config(text=file_name)

    def on_single_click(self, event):
        self.process_image_click(0)

    def on_single_click_canvas1(self, event):
        self.process_image_click(0)

    def on_single_click_canvas2(self, event):
        self.process_image_click(1)

    def extract_file_name(self, path):
        return os.path.basename(path)

            
    def display_metadata(self, metadata):
        # Очищаем текстовое поле перед отображением новых данных
        self.metadata_text.delete(1.0, tk.END)

        # Если метаданные существуют, форматируем их для отображения
        if metadata:
            metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
        else:
            metadata_text = ""

        # Вставляем текст в текстовое поле
        self.metadata_text.insert(tk.END, metadata_text)
        self.metadata_text.yview_moveto(0)  # Прокручиваем текст к началу

    def on_close(self):
        self.first_image_displayed = False
        self.current_image = None
        self.current_image_path = None
        self.current_images = [None, None]
        self.current_image_paths = [None, None]
        self.top.destroy()

    def open_results_folder(self):
        if os.path.exists(self.results_folder):
            subprocess.Popen(f'explorer "{os.path.abspath(self.results_folder)}"')
        else:
            print(f"Results folder '{self.results_folder}' not found")

    def remove_results(self):
        # Создаем диалог для удаления файлов
        dialog = FileDeletionDialog([self.results_folder])

        # Ждем, пока диалог не будет закрыт
        dialog.wait_window(dialog)

        # Проверяем, было ли выполнено удаление
        if dialog.deletion_done:
            self.on_close()


# Correspondence of data types and file extensions
data_types = {
    "region_data": "_locations",
    "stack_image": "_stack",
    "target_channel_image": "_synaptotag",
    "region_labels_image": "_with_roi",
    "filtered_image": "_denoised",
    "binary_masks_image": "_masks_roi_crop",
    "binary_mask": "_roi_mask",
    "binary_result_table": "_roi_result_table",
    "binary_summary_table": "_summary_roi_result_table",
    "histogram_image": "_hist",
    "histogram_data": "_histograms"
}
# Helper function to filter files by allowed data types
def filter_files_by_type(allowed_types, file_list):
    allowed_extensions = [data_types[dt] for dt in allowed_types]
    filtered_files = [file for file in file_list if any(ext in file for ext in allowed_extensions)]
    return filtered_files


def get_region_names(excel_paths):
    region_names = set()  # Используем множество для автоматического удаления дубликатов

    for excel_path in excel_paths:
        # Check if the Excel file exists
        if os.path.exists(excel_path):
            try:
                # Load the Excel file and sheet
                coords_df = pd.read_excel(excel_path, sheet_name='ROI_Coordinates')
                
                # Loop through every second column, assuming these are '_x' columns
                for col_x in coords_df.columns[::2]:  # Take every second column as '_x'
                    if '_x' in col_x:  # Ensure the column has '_x' in its name
                        location_name = col_x.rsplit('_', 1)[0]  # Extract the location name
                        region_names.add(location_name)  # Add to set to ensure uniqueness

            except Exception as e:
                print(f"Error loading regions from {excel_path}: {e}")
        else:
            print(f"Excel file does not exist: {excel_path}")

    # Return the unique region names as a sorted list
    return sorted(region_names)

# Example usage:
# excel_paths = ["/path/to/file1.xlsx", "/path/to/file2.xlsx"]
# region_names = get_region_names(excel_paths)
# print(region_names)



def delete_selected_regions(excel_paths, selected_regions):
    for excel_path in excel_paths:
        # Check if the Excel file exists
        if os.path.exists(excel_path):
            try:
                # Load the Excel file and sheet
                coords_df = pd.read_excel(excel_path, sheet_name='ROI_Coordinates')

                # Find columns to drop based on the selected regions
                columns_to_drop = []
                for region in selected_regions:
                    col_x = f"{region}_x"
                    col_y = f"{region}_y"
                    if col_x in coords_df.columns and col_y in coords_df.columns:
                        columns_to_drop.extend([col_x, col_y])  # Mark both '_x' and '_y' columns for deletion

                if columns_to_drop:
                    # Drop the selected columns
                    coords_df.drop(columns=columns_to_drop, inplace=True)

                    # Check if there are any remaining region columns (_x and _y)
                    remaining_region_columns = [
                        col for col in coords_df.columns if col.endswith('_x') or col.endswith('_y')
                    ]

                    if not remaining_region_columns:
                        # If no region columns are left, delete the Excel file
                        os.remove(excel_path)
                        print(f"Deleted file {excel_path} because no regions are left.")
                    else:
                        # Save the updated DataFrame back to the same Excel file
                        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                            coords_df.to_excel(writer, sheet_name='ROI_Coordinates', index=False)
                        print(f"Deleted regions {', '.join(selected_regions)} from {excel_path}")

                else:
                    print(f"No matching regions found in {excel_path}")

            except Exception as e:
                print(f"Error processing {excel_path}: {e}")
        else:
            print(f"Excel file does not exist: {excel_path}")

# File deletion dialog window
class FileDeletionDialog(tk.Toplevel):
    def __init__(self, folders):
        super().__init__()
        self.title("Delete Data")
        # Центрируем окно
        self.geometry("600x600")
        self.resizable(False, False)
        self.iconbitmap(icon_path)
        
        self.folders = folders
        self.deletion_done = False  # Track if deletion happened
        self.all_selected = False  # Добавляем атрибут для отслеживания состояния кнопки

        # Store checkbox states for each data type
        self.check_vars = {dt: tk.BooleanVar(value=False) for dt in data_types}  # Initially all unchecked
        self.region_check_var = tk.BooleanVar(value=False)  # Check state for "Region Data"
        self.all_regions_var = tk.BooleanVar(value=True)  # Check state for "all regions"
        self.all_files = self.get_all_files_in_folders()
        self.region_coordinate_files = filter_files_by_type(['region_data'], self.all_files)
        
        
        # Создаем кнопку Select All / Deselect All
        self.select_button = Button(self, text="Select All", command=self.toggle_select_all)
        self.select_button.grid(row=0, column=0, pady=10)

        # Create checkboxes for each data type
        Checkbutton(self, text="Region Data", variable=self.region_check_var, onvalue=True, offvalue=False, command=self.toggle_region_options).grid(row=1, column=0, sticky="w")
        row = 2  # Start from row 1 for other data type checkboxes
        for dt, var in self.check_vars.items():
            if dt != "region_data":  # Skip "Region Data" since we handle it separately
                label = dt.replace("_", " ").title()  # Format label
                Checkbutton(self, text=label, variable=var, onvalue=True, offvalue=False).grid(row=row, column=0, sticky="w")
                row += 1

        # Create a Canvas to hold the "all regions" checkbox and the region list
        self.canvas = Canvas(self, highlightthickness=0)
        self.all_regions_cb = Checkbutton(self.canvas, text="all regions", variable=self.all_regions_var, onvalue=True, offvalue=False, command=self.toggle_region_list)
        self.all_regions_cb.grid(row=0, column=0, sticky="w")
        
        self.region_listbox = None
        self.scrollbar = None
        
        # Add the canvas, but hide it for now
        self.canvas.grid(row=1, column=1, sticky="w", pady=10)
        self.canvas.grid_remove()  # Hide it initially

        # Button to start deletion process
        Button(self, text="Delete Selected Data", command=self.confirm_delete).grid(row=row + 1, column=0, pady=10)

        # Применяем тему через ThemeManager
        theme_manager = ThemeManager()
        theme_manager.apply_theme(self)
        
    def toggle_select_all(self):
        """Функция для переключения состояния выбора всех флажков"""
        if self.all_selected:
            # Если все выбраны, то снимаем все флажки
            for var in self.check_vars.values():
                var.set(False)
            
            # Снимаем флажок с "Region Data" и скрываем canvas
            self.region_check_var.set(False)
            self.all_regions_var.set(False)
            self.canvas.grid_remove()  # Скрываем canvas
            
            # Обновляем текст кнопки на "Select All"
            self.select_button.config(text="Select All")
        else:
            # Если не выбрано, выбираем все флажки
            for var in self.check_vars.values():
                var.set(True)
            
            # Выбираем "Region Data" и ставим галочку на "all regions"
            self.region_check_var.set(True)
            self.all_regions_var.set(True)
            self.canvas.grid()  # Показываем canvas
            self.hide_region_listbox()  # Убираем список регионов, если был показан
            
            # Обновляем текст кнопки на "Deselect All"
            self.select_button.config(text="Deselect All")
        
        # Переключаем состояние
        self.all_selected = not self.all_selected

    def toggle_region_options(self):
        """ Show or hide the canvas with 'all regions' and listbox based on 'Region Data' selection """
        if self.region_check_var.get():
            self.canvas.grid()  # Show the canvas when Region Data is selected
            self.all_regions_var.set(True)  # Reset "all regions" checkbox to True
            self.hide_region_listbox()  # Hide the listbox if visible
        else:
            self.canvas.grid_remove()  # Hide the canvas when Region Data is unselected

    def toggle_region_list(self):
        """ Show or hide the region list based on 'all regions' checkbox """
        if not self.all_regions_var.get():
            self.show_region_listbox()
        else:
            self.hide_region_listbox()

    def show_region_listbox(self):
        """ Display the listbox for selecting specific regions inside the canvas """
        if not self.region_listbox:
            self.scrollbar = Scrollbar(self.canvas)
            self.scrollbar.grid(row=1, column=1, sticky="ns")
            self.region_listbox = Listbox(self.canvas, selectmode="multiple", yscrollcommand=self.scrollbar.set)
            self.region_listbox.grid(row=1, column=0, sticky="w")
            self.scrollbar.config(command=self.region_listbox.yview)
            
            # Add fictional region names to the listbox
            for region in get_region_names(self.region_coordinate_files):
                self.region_listbox.insert("end", region)
        else:
            self.region_listbox.grid(row=1, column=0, sticky="w")
            self.scrollbar.grid(row=1, column=1, sticky="ns")

    def hide_region_listbox(self):
        """ Hide the listbox and its scrollbar """
        if self.region_listbox:
            self.region_listbox.grid_remove()
        if self.scrollbar:
            self.scrollbar.grid_remove()

    def confirm_delete(self):
        """ Ask the user to confirm deletion and delete selected files/regions """
        # List of selected data types
        selected_data_types = [dt for dt, var in self.check_vars.items() if var.get()]
        
        if self.region_check_var.get():
            selected_data_types.append("region_data")
        
        if not selected_data_types:
            messagebox.showwarning("Error", "No data types selected for deletion.")
            return
        
        # Confirmation dialog before deleting
        answer = messagebox.askyesno(
            "Delete Confirmation",
            "Are you sure you want to delete selected data?"
        )
        if answer:
            self.delete_files(selected_data_types)

    def delete_files(self, selected_data_types):
        
        files_to_delete = filter_files_by_type(selected_data_types, self.all_files)
        self.deletion_done = True  # Mark deletion as done
        
        for file in files_to_delete:
            if data_types["region_data"] in file:
                # Handle region-specific deletion if "all regions" is not selected
                if not self.all_regions_var.get() and self.region_listbox:
                    selected_regions = [self.region_listbox.get(i) for i in self.region_listbox.curselection()]
                    if selected_regions:
                        delete_selected_regions([file], selected_regions)
                        continue
            try:
                os.remove(file)
                print(f"Deleted file: {file}")
            except Exception as e:
                print(f"Error deleting file {file}: {e}")
        
        messagebox.showinfo("Deletion Complete", "The selected data have been deleted.")
        self.destroy()

    def get_all_files_in_folders(self):
        all_files = []
        for folder in self.folders:
            if not os.path.exists(folder):
                print(f"Folder does not exist, skipping: {folder}")
                continue
            for dirpath, _, filenames in os.walk(folder):
                for file in filenames:
                    all_files.append(os.path.join(dirpath, file))
        return all_files

# Example usage:
# dialog = FileDeletionDialog(folders=["/path/to/folder1", "/path/to/folder2"])
# dialog.mainloop()

def create_metadata_text_field(parent_frame, width=50, height=5):
    """
    Creates a text box to display metadata with vertical scrolling.

    :param parent_frame: the parent frame in which the text field will be placed
    :param width: width of the text field in characters
    :param height: height of the text field in lines
    :return: tuple (text field, scrollbar)
    """
    # Создаем текстовое поле
    metadata_text = tk.Text(
        parent_frame,
        width=width,   # Ширина поля в символах
        height=height,  # Высота поля в строках
        wrap='word'  # Перенос слов по словам, чтобы текст не выходил за рамки
    )
    #metadata_text.pack(side=tk.LEFT, padx=10, pady=5)

    # Создаем вертикальный скроллбар
    metadata_scrollbar = tk.Scrollbar(
        parent_frame,
        orient=tk.VERTICAL,
        command=metadata_text.yview
    )
    #metadata_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

    # Привязываем скроллбар к текстовому полю
    metadata_text.config(yscrollcommand=metadata_scrollbar.set)

    # Возвращаем текстовое поле и скроллбар
    return metadata_text, metadata_scrollbar

   

class ThemeManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ThemeManager, cls).__new__(cls)
            cls._instance.style = None  # Сюда будет сохраняться стиль
        return cls._instance

    def apply_theme(self, widget):
        if self.style:
            theme_background = self.style.lookup('TFrame', 'background')
            theme_font = self.style.lookup('TLabel', 'font')
            theme_foreground = self.style.lookup('TLabel', 'foreground')

            def update_widget_appearance(w):
                try:
                    w.configure(bg=theme_background)
                except tk.TclError:
                    pass

                try:
                    w.configure(font=theme_font, fg=theme_foreground)
                except (tk.TclError, AttributeError):
                    pass

                for child in w.winfo_children():
                    update_widget_appearance(child)

            update_widget_appearance(widget)

    def set_style(self, style):
        self.style = style

    def get_theme_background(self):
        """Метод для получения фонового цвета темы."""
        if self.style:
            return self.style.lookup('TFrame', 'background')
        return None  # Вернем None, если стиль не установлен

    def get_theme_highlight(self):
        """Метод для получения цвета подсветки темы (focuscolor), либо инверсии заднего фона."""
        if self.style:
            # Пытаемся получить цвет подсветки (focuscolor)
            highlight_color = self.style.lookup('TButton', 'focuscolor')
            if highlight_color:
                return highlight_color
            
            # Если цвет подсветки не найден, пытаемся получить цвет фона и вернуть его инверсию
            background_color = self.get_theme_background()
            if background_color:
                return self.invert_color(background_color)
            
        return "#FFFFFF"  # Возвращаем белый по умолчанию, если ничего не найдено

    @staticmethod
    def invert_color(color):
        try:
            color = color.lstrip('#')
            if len(color) != 6:
                return color
            rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
            inverted_rgb = tuple(255 - c for c in rgb)
            return '#{:02x}{:02x}{:02x}'.format(*inverted_rgb)
        except (ValueError, TypeError):
            return color


import tkinter as tk

def initialize_window(root, title, width, height, resizable=False, icon_path=None):
    """
    Функция для инициализации окна с общими параметрами.
    
    :param root: Родительское окно (обычно `Tk()` или `Toplevel()`).
    :param title: Заголовок окна.
    :param width: Ширина окна.
    :param height: Высота окна.
    :param resizable: Если False, окно нельзя изменять по размеру.
    :param icon_path: Путь к иконке окна, если есть.
    :return: Инициализированное окно.
    """
    # Создаем новое окно
    window = tk.Toplevel(root)
    window.title(title)
    
    # Получаем размер экрана
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    # Вычисляем координаты для центрирования окна
    x = (screen_width // 2) - (width // 2)
    y = (screen_height // 2) - (height // 2)
    
    # Устанавливаем размер и положение окна
    window.geometry(f"{width}x{height}+{x}+{y}")
    
    # Запрещаем изменение размеров окна, если указано
    if not resizable:
        window.resizable(False, False)
    
    # Если передан путь к иконке, устанавливаем ее
    if icon_path:
        window.iconbitmap(icon_path)
    
    return window

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from readlif.reader import LifFile
import numpy as np
from PIL import Image
import os
import platform
import subprocess

class LifFileConversion(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Convert LIF Image")
        # Center the window
        self.geometry("300x300")
        self.resizable(False, False)
        self.center_window()

        # Initialize variables
        self.file_paths = []
        self.target_ch = tk.IntVar(value=0)
        self.output_dirs = set()  # Set to store output directories

        # Create UI elements
        self.create_widgets()

    def center_window(self):
        self.update_idletasks()
        width = 300
        height = 300
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        # Button to select files
        select_files_button = ttk.Button(self, text="Select LIF Files", command=self.select_files)
        select_files_button.pack(pady=10)

        # Entry to select target channel
        target_ch_label = ttk.Label(self, text="Target Channel:")
        target_ch_label.pack()
        target_ch_spinbox = ttk.Spinbox(self, from_=0, to=10, textvariable=self.target_ch)
        target_ch_spinbox.pack(pady=5)

        # Button to start conversion
        convert_button = ttk.Button(self, text="Start Conversion", command=self.start_conversion)
        convert_button.pack(pady=10)

        # Progress bars without labels
        progress_frame = ttk.Frame(self)
        progress_frame.pack(pady=10)

        self.file_progress = ttk.Progressbar(progress_frame, orient='horizontal', length=200, mode='determinate')
        self.file_progress.pack(pady=5)

        self.image_progress = ttk.Progressbar(progress_frame, orient='horizontal', length=200, mode='determinate')
        self.image_progress.pack(pady=5)

    def select_files(self):
        file_paths = filedialog.askopenfilenames(title="Select LIF Files", filetypes=[("LIF files", "*.lif")])
        if file_paths:
            self.file_paths = self.tk.splitlist(file_paths)
            # Bring the main window to the front after file selection
            self.lift()
            self.focus_force()
            # Message about the number of selected files has been removed

    def start_conversion(self):
        if not self.file_paths:
            messagebox.showwarning("No Files Selected", "Please select LIF files to convert.")
            return
        target_ch = self.target_ch.get()
        if target_ch < 0:
            messagebox.showwarning("Invalid Channel", "Please enter a valid target channel number.")
            return

        # Initialize progress bars
        self.file_progress['maximum'] = len(self.file_paths)
        self.file_progress['value'] = 0
        self.image_progress['value'] = 0
        self.update_idletasks()

        # Clear the output directories set
        self.output_dirs.clear()

        # Start conversion process
        for file_index, file_path in enumerate(self.file_paths, start=1):
            try:
                self.convert_lif_to_tif(file_path, target_ch)
            except Exception as e:
                messagebox.showerror("Conversion Error", f"An error occurred while converting {os.path.basename(file_path)}:\n{e}")
            # Update file progress bar
            self.file_progress['value'] = file_index
            self.update_idletasks()

        # After conversion, offer to open the output directories
        if messagebox.askyesno("Conversion Complete", "All files have been converted.\nDo you want to open the folder(s) containing the results?"):
            for directory in self.output_dirs:
                self.open_folder(directory)

    def convert_lif_to_tif(self, file_path, target_ch):
        lif = LifFile(file_path)
        img_list = [i for i in lif.get_iter_image()]
        # Initialize image progress bar for this file
        self.image_progress['maximum'] = len(img_list)
        self.image_progress['value'] = 0
        self.update_idletasks()

        for im_index, im_in in enumerate(img_list, start=1):
            z_list = [i for i in im_in.get_iter_z(t=0, c=0)]
            z_n = len(z_list)  # Number of depths
            channel_list = [i for i in im_in.get_iter_c(t=0, z=0)]
            ch_n = len(channel_list)  # Number of channels

            frames = self.collect_lif_frames(im_in, target_ch, z_n, ch_n)
            # Save frames to multi-layered TIFF
            output_path = self.get_output_path(file_path, target_ch, im_index)
            self.save_frames_as_tiff(frames, output_path)
            # Add the output directory to the set
            output_dir = os.path.dirname(output_path)
            self.output_dirs.add(output_dir)
            # Update image progress bar
            self.image_progress['value'] = im_index
            self.update_idletasks()

    def collect_lif_frames(self, im_in, ch, z_n, ch_n):
        def remove_shift(lst, ch):
            shift = ((ch_n - 1) * ch - 1) % len(lst)
            reverse_shift = len(lst) - shift
            return lst[reverse_shift:] + lst[:reverse_shift]

        ch = (ch_n - ch)

        frames_out = []
        for z_real in list(range(z_n)):
            z = (z_real * ch_n) % z_n
            c = (z % ch_n + ch) % ch_n
            frames_out.append(np.array(im_in.get_frame(z=z, c=c)))
        frames_out = np.array(remove_shift(frames_out, ch))

        return frames_out

    def get_output_path(self, file_path, target_ch, im_index):
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}ch{target_ch}im{im_index}.tif"
        return output_path

    def save_frames_as_tiff(self, frames, output_path):
        # Convert frames to PIL Images and save as multi-page TIFF
        frames = frames.astype(np.uint16)  # Ensure frames are in appropriate format
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(output_path, save_all=True, append_images=images[1:], compression="tiff_deflate")

    def open_folder(self, path):
        # Open the folder in the file explorer
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", path])
        else:  # Linux and other systems
            subprocess.Popen(["xdg-open", path])

def run_lif_file_conversion_dialog():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    app = LifFileConversion()
    app.mainloop()

# Установка иконки окна
current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(current_dir, "images", "synaptocatcher.ico")

