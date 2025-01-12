#%matplotlib qt
#%%
from tkinter import filedialog, messagebox, PhotoImage, Toplevel, scrolledtext, StringVar, BooleanVar, Tk, Canvas, Menu
from tkinter.ttk import Button, Label, Entry, OptionMenu, Style, Checkbutton, Separator, Combobox, Frame
from ttkthemes import ThemedTk
import tkinter as tk
import sys
import os
import json
import tempfile
import pandas as pd
import time
from os.path import splitext, basename, dirname, join, exists
from tqdm import tqdm
import threading
from threading import Thread
import queue
import itertools
import webbrowser
import subprocess
import datetime
import re
import asyncio
import traceback

# Adding current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = current_dir + "\\images\\synaptocatcher.ico"
sys.path.append(current_dir)
from image_processor import binarize_images, select_location, stack_image
from image_processor import filter_after_roi_selection, pp_one, define_hist
from statistics_processor import analyze_and_plot_many_graphs, analyze_and_plot_one_graph
from graphical_processor import ExperimentWindow, process_synCatch_image, ThumbnailViewer, ThemeManager, initialize_window
from graphical_processor import run_lif_file_conversion_dialog

TEMP_FILE = os.path.join(tempfile.gettempdir(), 'synapto_catch_params.json')

class RedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Automatically scrolls the text field down
    def flush(self):  # Need for compatibility with sys.stdout
        pass

class ROIAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Synapto Catcher")
        self.root.iconbitmap(icon_path)

        self.icons_path = {
            "binarize": current_dir + "\\images\\buttons\\binarize_32.png",
            "mark_area": current_dir + "\\images\\buttons\\mark_area_32.png",
            "histogram": current_dir + "\\images\\buttons\\histogram_32.png",
            "preview": current_dir + "\\images\\buttons\\preview_32.png",
            "final_analysis": current_dir + "\\images\\buttons\\final_analysis_32.png",
            "clear_area": current_dir + "\\images\\buttons\\clear_area_32.png"
        }

        self._update_in_progress = False  # Флаг для предотвращения повторных вызовов
        
        # Создаем два Canvas
        self.canvas1 = Canvas(root, width=500, height=750, highlightthickness=0)
        self.canvas2 = Canvas(root, width=500, height=750, highlightthickness=0)

        # Размещаем Canvas рядом друг с другом
        self.canvas2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.canvas1.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Load previous parameters if they exist
        self.params = self.load_params()
        #self.clear_param('location_names')
        #self.clear_param('last_location')

        # Create the main menu
        self.menu_bar = Menu(self.root)
        # Create the "File" menu
        self.file_menu = Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Exit", command=self.close_window)

        # Create the "View" menu
        self.view_menu = Menu(self.menu_bar, tearoff=0)

        # Переменные состояния для галочек, загружаем их значения из файла
        self.canvas1_visible = tk.BooleanVar(value=self.params.get('canvas1_visible', True))
        self.inner_canvas3_visible = tk.BooleanVar(value=self.params.get('inner_canvas3_visible', True))


        self.view_menu.add_checkbutton(label="Settings",
                                       onvalue=True,
                                       offvalue=False,
                                       variable=self.canvas1_visible,
                                       command=lambda: self.toggle_canvas(self.canvas1, self.canvas1_visible, 'canvas1_visible'))
        self.view_menu.add_checkbutton(label="Console",
                                       onvalue=True,
                                       offvalue=False,
                                       variable=self.inner_canvas3_visible,
                                       command=lambda: self.toggle_canvas(self.inner_canvas3, self.inner_canvas3_visible, 'inner_canvas3_visible'))



        # Create the "Edit" menu
        self.edit_menu = Menu(self.menu_bar, tearoff=0)
        self.edit_menu.add_command(label="Create table", command=self.create_protocol)
        self.edit_menu.add_command(label="Add files to table", command=self.add_files_to_protocol)
        self.edit_menu.add_command(label="Convert LIF image", command=run_lif_file_conversion_dialog)

        # Create the "Statistics" menu
        self.stats_menu = Menu(self.menu_bar, tearoff=0)
        self.stats_menu.add_command(label="Group analysis", command=self.age_group_analysis_window)
        self.stats_menu.add_command(label="Histogram analysis", command=self.define_hist_action)

        # Add elements to the main menu
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)
        self.menu_bar.add_cascade(label="Statistics", menu=self.stats_menu)
        self.menu_bar.add_cascade(label="View", menu=self.view_menu)

        # Set the main menu in the window
        self.root.config(menu=self.menu_bar)

        self.create_header(self.canvas1, "Settings", font_size=10, underline=True)
        self.create_separator(self.canvas1)

        self.create_label_and_entry(self.canvas1,"slice start:", default_value='2', attr_name="slice_start")
        self.create_label_and_entry(self.canvas1,"slice end:", default_value='6', attr_name="slice_end")
        self.create_label_and_checkbutton(
            self.canvas1, 
            "select all slices", 
            default_value=False, 
            attr_name='all_slices_selected', 
            on_change=self.on_all_slices_selected_change
        )
        self.create_label_and_entry(self.canvas1,"target channel:", default_value='1', attr_name="target_ch")
        self.create_label_and_entry(self.canvas1,"additional channel:", default_value='4', attr_name="second_ch")
        self.create_label_and_entry(self.canvas1,"pixel_to_micron_ratio:", default_value='0.141', attr_name="pixel_to_micron_ratio")

        self.pixel_to_micron_ratio = float(self.pixel_to_micron_ratio_entry.get())

        self.create_separator(self.canvas1)
        self.create_label_and_entry(self.canvas1,"Filter radius:", default_value=17, attr_name="filter_radius")
        #self.create_button(self.canvas1,"2. Filter", self.filter_action)

        self.create_separator(self.canvas1)
        self.create_label_and_option_menu(self.canvas1,"Binarization Method:", ['otsu', 'otsu', 'max_entropy', 'yen', 'li', 'isodata', 'mean', 'minimum'], default_value='otsu', attr_name="binarization")
        self.create_label_and_entry(self.canvas1,"Min size of an object:", default_value=20, attr_name="min_size")
        self.create_label_and_entry(self.canvas1,"Max size of an object:", default_value=200, attr_name="max_size")

        #self.create_button(self.canvas1,"Remove bad spots", self.remove_ccp_action)

        # Elements for postprocess
        self.create_separator(self.canvas1)

        #self.create_button(self.canvas1,"4. Combine images", self.combine_images_action)

        self.create_label_and_entry(self.canvas1,"Output Directory:", self.browse_output_dir, attr_name="output_dir")

        # CANVAS 2
        #self.create_button(self.canvas2,"Mark Location", self.select_location_action)
        self.create_label_and_entry(self.canvas2,"Select File:", self.browse_protocol, readonly=True, attr_name="protocol")
        self.create_label_and_entry(self.canvas2,"ID number:", default_value='all', attr_name="rows", on_change=self.update_thumbnail_action)
        self.create_separator(self.canvas2)

        #self.create_button(self.canvas2, "Preview", self.update_thumbnail_action,
        #            icon_path=self.icons_path["preview"])
        #self.inner_canvas2 = Canvas(self.canvas2, height=150, background='lightgray')
        #self.inner_canvas2.pack(fill=tk.BOTH, expand=True)
        self.inner_frame2 = Frame(self.canvas2)
        self.inner_frame2.pack(fill=tk.BOTH, expand=True)
        


        self.create_separator(self.canvas2)

        self.inner_canvas1 = Canvas(self.canvas2, height=150, highlightthickness=0)
        self.inner_canvas1.pack(fill=tk.BOTH, expand=True)
        self.create_button(self.inner_canvas1,"1. Mark region", self.select_location_action,
                           icon_path=self.icons_path["mark_area"], side='left', attr_name = "mark_area")
        self.create_button(self.inner_canvas1,"2. Filter and Binarize", self.binarize_action,
                           icon_path=self.icons_path["binarize"], side='left', attr_name = "binarize")
        self.create_button(self.inner_canvas1,"3. Histogram analysis", self.define_hist_action,
                           icon_path=self.icons_path["histogram"], side='left', attr_name = "histogram")
        self.create_button(self.inner_canvas1,"4. General analysis", self.run_postprocess,
                           icon_path=self.icons_path["final_analysis"], side='left', attr_name = "final_analysis")

        self.create_separator(self.canvas2)

        self.inner_canvas3 = Canvas(self.canvas2, highlightthickness=0)
        self.inner_canvas3.pack(fill=tk.BOTH, expand=True)
        # Create text area for logs and redirect stdout
        self.create_text_area_with_redirect(self.inner_canvas3)

        self.previous_percentage = -1

        self.total_symbols = 20
        self.printed_symbols = 0
        self.czi_thumbnails = None
        self.selected_ids = []
        
        print('Welcome to Synapto Catcher!')
        self.add_hyperlink("README", "https://github.com/ta3map/Synapto_Catcher/tree/main?tab=readme-ov-file#synapto-catcher-user-guide")
        print('Parameters will be saved in ' + TEMP_FILE)
        
        self.save_all_params()# после успешного считывания и/или заполнения params сохраняется
        self.check_file_type()
        self.update_thumbnail_action()

        # Показываем или скрываем канвасы в зависимости от загруженных параметров
        self.update_canvas_visibility([
            (self.canvas1, 'canvas1_visible'),
            (self.inner_canvas3, 'inner_canvas3_visible')
        ])
        
        # Проверяем скрытие настроек каналов
        self.on_all_slices_selected_change()

    def update_canvas_visibility(self, canvas_list):
        """
        Обновляет видимость канвасов на основе параметров.
        canvas_list: список кортежей (канвас, ключ состояния).
        """
        for canvas, key in canvas_list:
            # Проверяем состояние видимости из параметров
            if self.params.get(key, True):
                canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            else:
                canvas.pack_forget()

    def toggle_canvas(self, parent, state_var, param_key):
        # Показываем или скрываем Canvas в зависимости от состояния переменной
        if state_var.get():  # Если переменная True - показать
            parent.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        else:  # Иначе скрыть
            parent.pack_forget()

        # Сохраняем текущее состояние канваса
        self.save_params(param_key, state_var.get())

    def age_group_analysis_window(self):
        if hasattr(self, 'age_group_window') and self.age_group_window.winfo_exists():
            self.age_group_window.deiconify()
            self.age_group_window.focus()
            return

        icon_path = current_dir + "\\images\\synaptocatcher.ico"
        self.age_group_window = initialize_window(self.root, "Group analysis", 500, 300, icon_path=icon_path)        

        
        # Add some content to the new window
        self.create_label_and_entry(self.age_group_window,"Select Result Table:", self.browse_result_table, readonly=True, attr_name="result_table")
        self.create_button(self.age_group_window,"Check parameters", self.print_unique_ages_and_locations)
        self.create_label_and_entry(self.age_group_window,"Groups:", default_value='', attr_name="unique_ages")
        self.create_label_and_entry(self.age_group_window,"Locations:", default_value='', attr_name="unique_locations")
        self.create_label_and_option_menu(self.age_group_window,"Data choise:", ['Total Area', 'Average Size', '%Area', 'Mean'], default_value='%Area', attr_name="data_choise")
        self.create_button(self.age_group_window,"Stats between groups", self.violin_statistics_many_action)
        self.create_button(self.age_group_window,"Stats between regions", self.violin_statistics_one_action)
        
        theme_manager = ThemeManager()
        theme_manager.apply_theme(self.age_group_window)
        
    def set_entry_var(self, attr_name, value):
        entry_var = getattr(self, f"selected_{attr_name}", None)
        if entry_var is not None:
            entry_var.set(value)

    def get_entry_var(self, attr_name):
        entry_var = getattr(self, f"selected_{attr_name}", None)
        if entry_var is not None:
            return entry_var.get()
        return None
    def get_selected_option(self, attr_name):
        # Get a reference to the StringVar associated with the OptionMenu
        var = getattr(self, f"{attr_name}_method", None)

        if var is not None:
            # Return the current value selected in the OptionMenu
            return var.get()
        else:
            raise AttributeError(f"No StringVar found for {attr_name}")

    def print_unique_ages_and_locations(self):

        file_path = self.selected_result_table.get()
        # Call the function to extract unique ages and locations
        unique_ages_list, unique_locations_list, column_names = self.extract_unique_parameters(file_path)

        # Format ages and locations into a string
        ages_str = ', '.join(map(str, unique_ages_list))
        locations_str = ', '.join(unique_locations_list)

        # Set the values ​​in the boxes
        self.set_entry_var("unique_ages", ages_str)
        self.set_entry_var("unique_locations", locations_str)

        # Set available options
        self.update_option_menu("data_choise", column_names)

        # Create the final text for output
        result_text = f"Unique Groups: {ages_str}\nUnique Locations: {locations_str}"

        print(result_text)

    def browse_result_table(self):
        result_table_path = filedialog.askopenfilename(filetypes=[("Result table file", "*.xlsx")])
        if result_table_path:
            result_table_path = result_table_path.replace('/', '\\')
            self.selected_result_table.set(result_table_path)
            self.save_params('result_table', result_table_path)

    def extract_number(age_str):
        # Find all the numbers in a string and combine them into one number
        numbers = re.findall(r'\d+', age_str)
        return int(numbers[0]) if numbers else None

    def extract_unique_parameters(self, file_path):
        # Load a table from an Excel file
        df = pd.read_excel(file_path)

        # Save column names to a list
        column_names = df.columns.tolist()

        # Extract the Group column and remove empty values
        ages = df['Group'].dropna()
        # Apply the function to all elements of the column and find unique values
        unique_ages = sorted(set(ages))

        # Retrieve the Location column and remove empty values
        locations = df['selected_location'].dropna()
        # Find unique locations
        unique_locations = sorted(set(locations))

        # Return unique values ​​and a list of columns
        return unique_ages, unique_locations, column_names

    def violin_statistics_many_action(self):

        ages_str = self.get_entry_var("unique_ages")
        locations_str = self.get_entry_var("unique_locations")
        table_file_path = self.get_entry_var("result_table")
        base_name = splitext(basename(table_file_path))[0]
        output_folder = join(dirname(table_file_path), f"{base_name}_statistics")
        selected_option = [self.get_selected_option("data_choise")]

        # Convert a text string with ages into a list of numbers
        ages_list = ages_str.split(', ')
        # Convert a text string with locations into a list of strings
        locations_list = locations_str.split(', ')
        # List of numeric parameters

        analyze_and_plot_many_graphs(
            file_path=table_file_path,
            output_folder= output_folder,
            groups=ages_list,
            locations=locations_list,
            numerical_parameters = selected_option
        )

        return

    def violin_statistics_one_action(self):

        ages_str = self.get_entry_var("unique_ages")
        locations_str = self.get_entry_var("unique_locations")
        table_file_path = self.get_entry_var("result_table")
        base_name = splitext(basename(table_file_path))[0]
        output_folder = join(dirname(table_file_path), f"{base_name}_statistics")
        selected_option = [self.get_selected_option("data_choise")]

        # Convert a text string with ages into a list of numbers
        ages_list = ages_str.split(', ')
        # Convert a text string with locations into a list of strings
        locations_list = locations_str.split(', ')
        # List of numeric parameters

        analyze_and_plot_one_graph(
            file_path=table_file_path,
            output_folder= output_folder,
            groups=ages_list,
            locations=locations_list,
            numerical_parameters = selected_option
        )

        return

    def close_window(self):
        self.root.destroy()

    def create_separator(self, parent):
        separator = Separator(parent, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=10)


    def create_header(self, parent,
                      text="header",
                      font_family="Arial",
                      font_size=None,
                      font_style=None,
                      underline=False,
                      align='left',
                      foreground=None,
                      background=None,
                      padx = 5,
                      pady = 5):

        # Determine text alignment based on the align parameter
        if align == 'left':
            anchor = 'w'  # Align left
        elif align == 'center':
            anchor = 'center'  # Align center
        elif align == 'right':
            anchor = 'e'  # Align right
        else:
            anchor = 'w'  # Default to left if invalid value is provided

        # Set the font style and include underline if requested
        font_styles = [font_style] if font_style else []
        if underline:
            font_styles.append('underline')

        # Set the font, default to system font if size or style is not provided
        font = (font_family, font_size if font_size else 10, ' '.join(font_styles))

        # Create the header label with optional foreground and background colors
        header = Label(parent, text=text, font=font, anchor=anchor)

        if foreground:
            header.config(foreground=foreground)
        if background:
            header.config(background=background)

        header.pack(fill='x', padx=padx, pady=pady)

    def create_text_area_with_redirect(self, parent):
        self.text_area = scrolledtext.ScrolledText(parent, wrap=tk.WORD, width=50, height=15)
        self.text_area.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        redirect_text = RedirectText(self.text_area)
        sys.stdout = redirect_text
        # button = Button(self.root, text="Print Test", command=self.on_button_click, width=20)
        # button.pack(padx=5, pady=5)


    def create_label_and_entry(self, parent, label_text, command=None, default_value='', readonly=False, attr_name=None, on_change=None):
        frame = Frame(parent)
        frame.pack(padx=5, pady=5, fill=tk.X)

        label = Label(frame, text=label_text, width=30)
        label.pack(side=tk.LEFT)

        entry_var = StringVar(value=self.params.setdefault(attr_name, default_value))
        #entry_var = StringVar(value=self.params.get(attr_name, default_value))
        entry = Entry(frame, textvariable=entry_var, width=20, state='readonly' if readonly else 'normal')
        entry.pack(side=tk.LEFT, padx=5)

        # Добавляем trace с вызовом save_params и пользовательской функции
        entry_var.trace_add("write", lambda *args: self.on_entry_change(attr_name, entry_var.get(), on_change))

        if command:
            button = Button(frame, text="...", command=command)
            button.pack(side=tk.LEFT, padx=5)

        if attr_name:
            setattr(self, f"{attr_name}_entry", entry)
            setattr(self, f"selected_{attr_name}", entry_var)
            setattr(self, f"{attr_name}_label", label)
            
    def create_label_and_checkbutton(self, parent, label_text, attr_name=None, on_change=None, default_value=False):
        frame = Frame(parent)
        frame.pack(padx=5, pady=5, fill=tk.X)

        label = Label(frame, text=label_text, width=30)
        label.pack(side=tk.LEFT)

        # Создаем переменную для состояния галочки с возможностью задать значение по умолчанию
        check_var = tk.BooleanVar(value=self.params.setdefault(attr_name, default_value))
        
        # Создаем галочку
        checkbutton = Checkbutton(frame, variable=check_var, command=lambda: self.on_entry_change(attr_name, check_var.get(), on_change))
        checkbutton.pack(side=tk.LEFT, padx=5)

        if attr_name:
            setattr(self, f"{attr_name}_checkbutton", checkbutton)
            setattr(self, f"selected_{attr_name}", check_var)
            setattr(self, f"{attr_name}_label", label)

            
    def on_entry_change(self, key, value, on_change=None, delay=1000):
        self.root.after(300, lambda: self.save_params(key, value))
        # Проверка на то, идет ли уже обновление
        if not self._update_in_progress:
            self._update_in_progress = True
            # Откладываем выполнение save_params и on_change
            self.root.after(delay, lambda: self._complete_change(key, value, on_change))

    def _complete_change(self, key, value, on_change):
        # Вызываем on_change, если оно передано
        if on_change:
            on_change()

        # Сбрасываем флаг
        self._update_in_progress = False

    def on_all_slices_selected_change(self):
        # Получаем значение галочки
        all_slices_selected = self.selected_all_slices_selected.get()
        
        # Скрываем или показываем slice_start и slice_end в зависимости от состояния галочки
        self.toggle_entry('slice_start', not all_slices_selected)
        self.toggle_entry('slice_end', not all_slices_selected)
        
    def create_button(self, parent, text, command, icon_path=None, side="top", attr_name=None):
        if icon_path:
            try:
                icon = PhotoImage(file=icon_path)
            except Exception as e:
                icon = None
        else:
            icon = None

        # Создаем кнопку с иконкой или без
        if icon:
            button = Button(parent, text=text, command=command, image=icon, compound="left")
            button.image = icon  # Сохранение иконки для предотвращения удаления сборщиком мусора
        else:
            button = Button(parent, text=text, command=command, width=20)

        button.pack(side=side, padx=5, pady=5)

        # Если передан attr_name, сохраняем кнопку и её параметры
        if attr_name:
            setattr(self, f"{attr_name}_button", button)  # Сохраняем ссылку на кнопку
            setattr(self, f"{attr_name}_button_side", side)  # Сохраняем информацию о side

    def toggle_button(self, attr_name, enabled):
        button = getattr(self, f"{attr_name}_button", None)
        side = getattr(self, f"{attr_name}_button_side", None)  # Получаем сохраненный side

        if button and side:
            if enabled:
                button.pack(side=side)  # Показываем кнопку с сохраненным side
            else:
                button.pack_forget()  # Скрываем кнопку
                
    def toggle_entry(self, attr_name, enabled):
        entry = getattr(self, f"{attr_name}_entry", None)
        entry_var = getattr(self, f"selected_{attr_name}", None)
        label = getattr(self, f"{attr_name}_label", None)
        if entry and entry_var:
            if enabled:
                label.pack(side=tk.LEFT)
                entry.pack(side=tk.LEFT)
                if entry_var.get() == '':
                    entry_var.set('all')
            else:
                label.pack_forget()
                entry.pack_forget()
                entry_var.set('all')
            
    # def toggle_entry(self, attr_name, enabled):
    #     entry = getattr(self, f"{attr_name}_entry", None)
    #     entry_var = getattr(self, f"selected_{attr_name}", None)
    #     label = getattr(self, f"{attr_name}_label", None)
    #     if entry and entry_var:
    #         if enabled:
    #             label.pack(side=tk.LEFT)
    #             entry.pack(side=tk.LEFT)
    #             # entry.config(state='normal')
    #             if entry_var.get()=='':
    #                 entry_var.set('all')
    #         else:
    #             label.pack_forget()
    #             entry.pack_forget()
    #             # entry.config(state='disabled')
    #             entry_var.set('')


    def create_label_and_option_menu(self, parent, label_text, options, default_value, attr_name):
        frame = Frame(parent)
        frame.pack(padx=5, pady=5, fill=tk.X)

        label = Label(frame, text=label_text, width=30)
        label.pack(side=tk.LEFT)

        var = StringVar(value=self.params.get(attr_name, default_value))
        option_menu = OptionMenu(frame, var, *options)
        option_menu.pack(side=tk.LEFT, padx=5)

        # Saving a link to the OptionMenu
        setattr(self, f"{attr_name}_option_menu", option_menu)

        var.trace_add("write", lambda *args: self.save_params(attr_name, var.get()))

        # Store a reference to StringVar to access the selected value
        setattr(self, f"{attr_name}_method", var)


    def update_option_menu(self, attr_name, new_options):
        option_menu = getattr(self, f"{attr_name}_option_menu")  # Access to OptionMenu
        menu = option_menu["menu"]  # Access a Menu object inside an OptionMenu

        # Clear current menu
        menu.delete(0, "end")

        # Adding new options
        for option in new_options:
            menu.add_command(label=option, command=lambda value=option: option_menu.setvar(option_menu.cget("textvariable"), value))

        # Set default value (first element of new list)
        if new_options:
            option_menu.setvar(option_menu.cget("textvariable"), new_options[0])


    def check_file_type(self):
        attr_name = 'protocol'
        protocol_path = self.params.get(attr_name, '')
        print('Main File:')
        print(protocol_path)

        # Checking for protocol extension
        if protocol_path.endswith('.xlsx'):
            self.toggle_entry("rows", True)
            self.toggle_button("final_analysis", True)
        else:
            self.toggle_entry("rows", False)
            self.toggle_button("final_analysis", False)

    def flatten_list(self, nested_list):
        flattened = list(itertools.chain.from_iterable(nested_list))
        return flattened

    def on_button_click(self):
        print("test")
        self.add_hyperlink("Click here for website", "https://www.example.com")
        self.add_file_link("Open File", "/path/to/your/file.txt")
        self.test_progress()

    def add_hyperlink(self, text, url):
        def open_url(event):
            webbrowser.open_new(url)
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.tag_add("hyperlink", "end-2c linestart", "end-2c lineend")
        self.text_area.tag_config("hyperlink", foreground="blue", underline=1)
        self.text_area.tag_bind("hyperlink", "<Button-1>", open_url)

    def add_file_link(self, text, file_path):
        tag_name = f"filelink_{file_path}"
        self.text_area.insert(tk.END, text + "\n")
        self.text_area.tag_add(tag_name, "end-2c linestart", "end-2c lineend")
        self.text_area.tag_config(tag_name, foreground="green", underline=1)
        self.text_area.tag_bind(tag_name, "<Button-1>", lambda event, fp=file_path: self.open_file(fp))

    def open_file(self, file_path):
        file_path = file_path.replace('/', '\\')
        if os.path.exists(file_path):
            try:
                # cmd = f'explorer /select,"{file_path}"'
                cmd = f'explorer "{file_path}"'
                #print(cmd)
                subprocess.Popen(cmd)
            except Exception as e:
                print(f"Failed to open file '{file_path}': {e}")
        else:
            print(f"File '{file_path}' not found")


    def add_file_links_from_list(self, files_out):
        files_out = self.flatten_list(files_out)
        for filepath in files_out:
            self.add_file_link(filepath, filepath)


    def browse_protocol(self):
        protocol_path = filedialog.askopenfilename(filetypes=[("Protocol files", "*.lif;*.tif;*.czi;*.xlsx")])
        if protocol_path:
            protocol_path = protocol_path.replace('/', '\\')
            self.selected_protocol.set(protocol_path)
            self.save_params('protocol', protocol_path)
            protocol_dir = os.path.dirname(protocol_path)
            default_output_dir = protocol_dir
            self.selected_output_dir.set(default_output_dir)
            self.save_params('output_dir', default_output_dir)

            self.check_file_type()
            self.update_thumbnail_action()
            #self.prepare_data()


    def update_progress_bar(self, value, maximum):
        percentage = value / maximum
        symbols_to_print = round(self.total_symbols * percentage)
        additional_symbols = symbols_to_print - self.printed_symbols

        if additional_symbols > 0:
            print('█' * additional_symbols, end='', flush=True)
            self.printed_symbols = symbols_to_print
        if value >= maximum-1:
            self.printed_symbols = 0
            print('', flush=False)
        #if value == 0:
        #    print("0%------50%------100%")

    def create_protocol(self):
        # Open a window to select files
        messagebox.showinfo("Select files", "Select files for the table.")
        filetypes = [("Image Files", "*.czi *.lif *.tif")]
        filepaths = filedialog.askopenfilenames(title="Select files", filetypes=filetypes)

        if not filepaths:
            print("No files selected")
            return

        # Create a table with the required columns
        data = {
            'filepath': filepaths,
            'comment': [''] * len(filepaths),
            'location': [''] * len(filepaths),
            'ID': list(range(1, len(filepaths) + 1)),
            'take_to_stat': [''] * len(filepaths),
            'Group': [''] * len(filepaths)
        }
        df = pd.DataFrame(data)

        # Open a window to select the path to save the table
        messagebox.showinfo("Select file", "Choose where to save the table.")
        # Get the current date and time
        current_datetime = datetime.datetime.now().strftime("%d.%m.%Y_%H.%M")
        # Form a file name with date and time
        default_filename = f"table_{current_datetime}.xlsx"
        xlsx_table_path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                               filetypes=[("Excel files", "*.xlsx")],
                                               initialfile=default_filename)

        if xlsx_table_path:
            # Saving a table to an Excel file
            df.to_excel(xlsx_table_path, index=False)

            # Open a file using Explorer
            self.open_file(xlsx_table_path)
        else:
            print("File save cancelled")


    def add_files_to_protocol(self):
        # Open a window to select an existing table
        messagebox.showinfo("Select file", "Select the table to update.")

        xlsx_table_path = filedialog.askopenfilename(title="Select Excel file to update", filetypes=[("Excel files", "*.xlsx")])

        if not xlsx_table_path:
            print("No Excel file selected")
            return

        required_columns = ['filepath', 'comment', 'location', 'ID', 'take_to_stat', 'Group']
        # Checking file type
        if xlsx_table_path.endswith('.xlsx'):
            df = pd.read_excel(xlsx_table_path)
            # Check if the DataFrame contains all the required columns
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                messagebox.showerror("Error", f"Selected table is missing required columns: {', '.join(missing_columns)}. Please choose a different file.")
                print('Error! Please choose a different file.')
                return

        # Read an existing table
        df_existing = pd.read_excel(xlsx_table_path)

        # Open a window to select files to add
        messagebox.showinfo("Select files", "Select files for the table.")
        filetypes = [("Image Files", "*.czi *.lif *.tif")]
        new_filepaths = filedialog.askopenfilenames(title="Select files to add", filetypes=filetypes)

        if not new_filepaths:
            print("No new files selected")
            return

        # Determine the starting number for new ID
        start_ID = df_existing['ID'].max() + 1 if not df_existing.empty else 1

        # Create a new part of the table
        new_data = {
            'filepath': new_filepaths,
            'comment': [''] * len(new_filepaths),
            'location': [''] * len(new_filepaths),
            'ID': list(range(start_ID, start_ID + len(new_filepaths))),
            'take_to_stat': [''] * len(new_filepaths),
            'Group': [''] * len(new_filepaths)
        }
        df_new = pd.DataFrame(new_data)

        # Merging existing and new tables
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # Saving the updated table
        df_combined.to_excel(xlsx_table_path, index=False)

        # Open a file using Explorer
        self.open_file(xlsx_table_path)


    def update_thumbnail_action(self):
        protocol_path = self.selected_protocol.get()

        if os.path.exists(protocol_path):
            try:
                df, rows_to_process = self.prepare_data_thumbnail()
                #print("update_thumbnail_action")
                images = df.loc[rows_to_process, 'filepath'].tolist()

                # Получаем ID и комментарии
                ids = df.loc[rows_to_process, 'ID']
                comments = df.loc[rows_to_process, 'comment']

                # Формируем список ID с комментариями
                IDs = [
                    f"ID: {str(id_val)}" + (f" | {str(comment)}" if pd.notna(comment) and str(comment).strip() else "")
                    for id_val, comment in zip(ids, comments)
                ]

                if hasattr(self, 'thumbnail_viewer'):
                    self.thumbnail_viewer.destroy()  # Уничтожаем предыдущий экземпляр

                # Создаем новый экземпляр ThumbnailViewer
                self.thumbnail_viewer = ThumbnailViewer(
                    parent=self.inner_frame2,
                    images=images,
                    comments=IDs,image_ids=ids,
                    on_selection_change = self.update_selected_ids,
                    open_image_func=process_synCatch_image,
                    on_double_click=ExperimentWindow,
                    max_per_page=10
                )

                # Устанавливаем фокус                
                self.thumbnail_viewer.thumbnail_inner_frame.focus_set()
                
            except Exception as e:
                print(f"Error: {e}")
        else:
            print('--------------------')
            print("ERROR:")
            print("Main file does not exist.")
            print('--------------------')
            

    def update_selected_ids(self, selected_ids):
        self.selected_ids = selected_ids
        

    def explore_results_action(self):
        # Подготовка данных
        df, rows_to_process = self.prepare_data()

        # Если в rows_to_process более одного эксперимента, предложим пользователю выбрать один
        if len(rows_to_process) > 1:
            # Получаем список доступных экспериментов для выбора
            experiments_choice = df.iloc[rows_to_process]['ID'].tolist()

            # Создаем маленькое окно для выбора эксперимента
            selection_window = tk.Tk()
            selection_window.title("Choose ID")

            # Создаем метку и выпадающий список для выбора эксперимента
            label = Label(selection_window, text="Select ID:")
            label.pack(pady=5)

            experiment_var = tk.StringVar()
            dropdown = Combobox(selection_window, textvariable=experiment_var)
            dropdown['values'] = experiments_choice
            dropdown.pack(pady=5)

            # Устанавливаем первый элемент как выбранный по умолчанию
            experiment_var.set(experiments_choice[0])
            dropdown.current(0)

            # Привязываем событие выбора из списка к переменной experiment_var
            def on_select(event):
                selected_experiment = dropdown.get()  # Получаем выбранное значение
                experiment_var.set(selected_experiment)  # Устанавливаем выбранное значение

            dropdown.bind("<<ComboboxSelected>>", on_select)  # Привязка события

            # Функция для обработки выбора эксперимента
            def on_experiment_select():
                selected_experiment = int(experiment_var.get())  # Получаем выбранный номер эксперимента
                selected_idx = df[df['ID'] == selected_experiment].index[0]  # Находим индекс выбранного эксперимента

                # Получаем путь к файлу
                file_path = df.iloc[selected_idx]['filepath']

                # Сбор комментария: берем все колонки, кроме 'ID' и 'filepath'
                comment_data = df.iloc[selected_idx].drop(['ID', 'filepath']).fillna('')
                comment = "\n".join([f"{col}: {val}" for col, val in comment_data.items()])

                # Закрываем окно выбора
                selection_window.destroy()

                # Открываем окно для выбранного эксперимента и передаем комментарий
                ExperimentWindow(file_path, comment=comment)

            # Кнопка для подтверждения выбора
            select_button = tk.Button(selection_window, text="Open ID", command=on_experiment_select)
            select_button.pack(pady=10)

            # Запуск окна выбора
            selection_window.mainloop()

        # Если эксперимент только один, сразу его обрабатываем
        elif len(rows_to_process) == 1:
            row_idx = rows_to_process[0]
            file_path = df.iloc[row_idx]['filepath']

            # Сбор комментария: берем все колонки, кроме 'ID' и 'filepath'
            comment_data = df.iloc[row_idx].drop(['ID', 'filepath']).fillna('')
            comment = "\n".join([f"{col}: {val}" for col, val in comment_data.items()])

            # Открываем окно для выбранного эксперимента с комментарием
            ExperimentWindow(file_path, comment=comment)

    def parse_entry(self, entry):
        # Если entry - число, уменьшаем его на 1, иначе оставляем как есть
        if entry.isdigit():
            entry = int(entry) - 1
        return entry
    
    def select_location_action(self):
        df, rows_to_process = self.prepare_data()
        total = len(rows_to_process)
        selected_location = ''
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            print(file_path)
            
            slice_start = self.parse_entry(self.slice_start_entry.get())
            slice_end = self.parse_entry(self.slice_end_entry.get())
            
            target_ch = int(self.target_ch_entry.get())-1
            dapi_ch = int(self.second_ch_entry.get())-1

            # Передаем значения в stack_image
            stack_image(file_path, slice_start, slice_end, target_ch, dapi_ch)
            
            coords_df, selected_location = select_location(file_path, self.root, initial_location=selected_location)
            
            if coords_df.empty:
                print("Processing stopped")
                break
            self.update_progress_bar(idx, total)


    async def async_filter_action(self):
        df, rows_to_process = self.prepare_data()
        filter_radius = int(self.filter_radius_entry.get())
        total = len(rows_to_process)
        files_out = []
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            location = df.iloc[row_idx]['location']
            files_out.append(filter_after_roi_selection(filter_radius, file_path, location))
            self.update_progress_bar(idx, total)
            await asyncio.sleep(0)  # Let's give management another task
        print("Images filtered successfully.")
        self.add_file_links_from_list(files_out)

    def filter_action(self):
        asyncio.run_coroutine_threadsafe(self.async_filter_action(), self.loop)

    def binarize_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            total = len(rows_to_process)
            filter_radius = int(self.filter_radius_entry.get())
            for idx, row_idx in enumerate(rows_to_process):
                file_path = df.iloc[row_idx]['filepath']
                row = df.iloc[row_idx]
                
                slice_start = self.parse_entry(self.slice_start_entry.get())
                slice_end = self.parse_entry(self.slice_end_entry.get())
                
                target_ch = int(self.target_ch_entry.get())-1
                dapi_ch = int(self.second_ch_entry.get())-1
            
                # Extraction from target channel
                stack_image(file_path, slice_start, slice_end, target_ch, dapi_ch)
                # Filtration
                filter_after_roi_selection(filter_radius, file_path)
                # Binarization
                binarize_images(
                    file_path, 
                    self.binarization_method.get(), int(self.min_size_entry.get()),
                    int(self.max_size_entry.get()), self.pixel_to_micron_ratio
                )

                self.update_progress_bar(idx, total)
                time.sleep(0)
            print("Binarization completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()


    def define_hist_action(self):
        df, rows_to_process = self.prepare_data()
        total = len(rows_to_process)
        files_out = []
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            location = df.iloc[row_idx]['location']
            slice_start = self.parse_entry(self.slice_start_entry.get())
            slice_end = self.parse_entry(self.slice_end_entry.get())
            target_ch = int(self.target_ch_entry.get())-1
            second_ch = int(self.second_ch_entry.get())-1
            print(file_path)
            #print(f"target_ch: {target_ch}")
            #print(f"second_ch: {second_ch}")
            files_out.append(define_hist(file_path, location, slice_start, slice_end, target_ch, second_ch, self.root))
            self.update_progress_bar(idx, total)
        print("all Histograms defined successfully.")

    # Асинхронная функция для выполнения постобработки
    async def async_run_postprocess(self):
        df, rows_to_process = self.prepare_data()
        output_directory = self.selected_output_dir.get()
        os.makedirs(output_directory, exist_ok=True)
        total = len(rows_to_process)

        summary_data_list = []
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            row = df.iloc[row_idx]           

            # Обрабатываем один файл
            summary_data_s = pp_one(file_path, row, output_directory)
            
            # Добавляем данные в общий список
            if summary_data_s:  # Добавляем только если данные есть                
                summary_data_list.extend(summary_data_s)
            
            # Обновляем прогресс
            self.update_progress_bar(idx, total)
            await asyncio.sleep(0)  # Отдаем управление другим задачам

        # Если удалось собрать данные, сохраняем их
        if summary_data_list:
            summary_df = pd.concat(summary_data_list, ignore_index=True)
            summary_df.drop(summary_df.columns[[0]], axis=1, inplace=True)
            summary_output_path = join(output_directory, "collected_roi_summary_data.xlsx")
            summary_df.to_excel(summary_output_path, index=False)
            
            print("General analysis completed successfully.")
            print("Analysis result:")
            self.add_file_links_from_list([[summary_output_path]])
        else:
            print("No valid summary data was collected.")

    def run_postprocess(self):
        asyncio.run_coroutine_threadsafe(self.async_run_postprocess(), self.loop)
        
    def prepare_data(self):
        protocol_path = self.selected_protocol.get()
        required_columns = ['filepath', 'comment', 'location', 'ID', 'take_to_stat', 'Group']

        # Checking file type
        if protocol_path.endswith('.xlsx'):
            df = pd.read_excel(protocol_path)
            IDs = self.selected_ids
            rows_to_process = df[df['ID'].isin(IDs)].index.tolist()
        else:
            # Create a DataFrame with the specified columns and values
            df = pd.DataFrame({
                'filepath': [protocol_path],
                'comment': [''],
                'location': ['unknown'],
                'ID': ['1'],
                'take_to_stat': [''],
                'Group': ['']
            })
            # Set rows_to_process to the first index (0)
            rows_to_process = [0]

        return df, rows_to_process
    
    def prepare_data_thumbnail(self):
        protocol_path = self.selected_protocol.get()
        required_columns = ['filepath', 'comment', 'location', 'ID', 'take_to_stat', 'Group']

        # Checking file type
        if protocol_path.endswith('.xlsx'):
            df = pd.read_excel(protocol_path)

            # Check if the DataFrame contains all the required columns
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                messagebox.showerror("Error", f"Selected table is missing required columns: {', '.join(missing_columns)}. Please choose a different file.")
                print('Error! Please choose a different file.')
                return None, None

            # Filter out rows where 'take_to_stat' is 'no' and log removed experiments
            removed_experiments = df[df['take_to_stat'] == 'no']['ID'].tolist()
            df = df[df['take_to_stat'] != 'no']
            #print(f"Removed IDs: {removed_experiments}")

            # Reset the indices of the DataFrame
            df.reset_index(drop=True, inplace=True)

            # Определяем максимальный ID из доступных данных
            max_id = df['ID'].max()

            # Получаем список экспериментов для обработки
            IDs = self.get_exps_to_process(max_id)

            # Find rows with the specified experiment numbers
            if IDs == 'all':
                rows_to_process = list(range(len(df)))
            else:
                rows_to_process = df[df['ID'].isin(IDs)].index.tolist()
        else:
            # Create a DataFrame with the specified columns and values
            df = pd.DataFrame({
                'filepath': [protocol_path],
                'comment': [''],
                'location': ['unknown'],
                'ID': ['1'],
                'take_to_stat': [''],
                'Group': ['']
            })
            # Set rows_to_process to the first index (0)
            rows_to_process = [0]

        return df, rows_to_process

    def get_exps_to_process(self, max_id):
        exps_input = self.rows_entry.get()
        if not self.selected_protocol.get() or not exps_input:
            raise ValueError("Please select a protocol and enter IDs to process.")
        return self.parse_exps(exps_input, max_id)

    def parse_exps(self, exps_input, max_id):
        exps_input = exps_input.strip().lower()
        if exps_input == 'all':
            return 'all'
        try:
            # Check if input is in the range format
            if ':' in exps_input:
                start, end = exps_input.split(':')
                start = int(start)

                # Обработка случая с end
                if end == 'end':
                    end = max_id
                else:
                    end = int(end)

                return list(range(start, end + 1))
            else:
                return [int(exp.strip()) for exp in exps_input.split(',')]
        except ValueError:
            raise ValueError("Invalid ID input. Enter comma-separated numbers, 'all', or a range in the format 'start:end'.")


    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.selected_output_dir.set(directory)
            self.save_params('output_dir', directory)

    def save_params(self, key, value):
        self.params[key] = value
        with open(TEMP_FILE, 'w') as f:
            json.dump(self.params, f)

    def save_all_params(self):
        with open(TEMP_FILE, 'w') as f:
            json.dump(self.params, f)
    
    def load_params(self):
        if os.path.exists(TEMP_FILE):
            try:
                with open(TEMP_FILE, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, KeyError):
                os.remove(TEMP_FILE)
        return {}

    def clear_param(self, key):
        # Проверяем, существует ли параметр с данным ключом
        if key in self.params:
            # Удаляем параметр
            del self.params[key]
            # Сохраняем обновлённые параметры
            with open(TEMP_FILE, 'w') as f:
                json.dump(self.params, f)

    def set_loop(self, loop):
        self.loop = loop

def start_event_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

try:
    import pyi_splash
    pyi_splash.update_text('UI Loaded ...')
    pyi_splash.close()
except:
    pass


# Функция для применения всех параметров темы
def apply_theme(root, style):
    # Получаем цвет фона и шрифт из текущей темы
    theme_background = style.lookup('TFrame', 'background')
    theme_font = style.lookup('TLabel', 'font')
    theme_foreground = style.lookup('TLabel', 'foreground')

    # Устанавливаем фон для главного окна
    root.configure(bg=theme_background)

    # Рекурсивная функция для применения фона, шрифта и цвета текста ко всем виджетам
    def update_widget_appearance(widget):
        try:
            widget.configure(bg=theme_background)
        except tk.TclError:
            pass  # Игнорируем виджеты, которые не поддерживают изменение фона

        try:
            widget.configure(font=theme_font, fg=theme_foreground)
        except (tk.TclError, AttributeError):
            pass  # Игнорируем виджеты, которые не поддерживают изменение шрифта или цвета текста

        for child in widget.winfo_children():
            update_widget_appearance(child)

    # Применяем изменения ко всем виджетам
    update_widget_appearance(root)

    # Попробуем изменить цвет заголовка окна (шапки)
    try:
        root.wm_attributes("-titlepath", theme_background)
    except tk.TclError:
        pass  # Изменение цвета заголовка не поддерживается на всех системах
    
if __name__ == "__main__":
    #root = tk.Tk()
    root = ThemedTk(theme="yaru")#yaru#breeze#Adapta
    root.configure(bg=root['background'])
    
    app = ROIAnalyzerApp(root)

    # Создаем объект стиля для работы с темой
    style = Style()
    
    # Настраиваем синглтон с темой
    theme_manager = ThemeManager()
    theme_manager.set_style(style)
    # Применяем тему к виджетам главного окна
    root.after(100, lambda: theme_manager.apply_theme(root))
    
    # Determine the screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Determine the window dimensions
    window_width = 1024
    window_height = 768

    # Calculate coordinates for window placement
    position_right = int(screen_width / 2 - window_width / 2)
    position_down = int(screen_height / 2 - window_height / 2)

    # Bring the window to the foreground
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    root.focus_force()
    # Set the size and position of the window
    root.geometry(f'{window_width}x{window_height}+{position_right}+{position_down}')
    root.wm_state("zoomed")
    # Create a new asyncio event loop and run it in a separate thread
    new_loop = asyncio.new_event_loop()
    t = Thread(target=start_event_loop, args=(new_loop,))
    t.start()

    app.set_loop(new_loop)

    root.mainloop()
    new_loop.call_soon_threadsafe(new_loop.stop)  # Stop the event loop when the window closes
