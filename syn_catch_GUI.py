from tkinter import filedialog, messagebox, Button, Label, Entry, StringVar, OptionMenu, Tk, ttk
import sys
import os
import json
import tempfile
import pandas as pd
import time
from os.path import splitext, basename, dirname, join, exists
from tqdm import tqdm
import threading
import queue
import itertools
from tkinter import scrolledtext
import webbrowser
import tkinter as tk

from tkinter import PhotoImage

import asyncio
from threading import Thread

import traceback
# %matplotlib qt

# Adding current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from roi_processor import process_file, binarize_images, remove_ccp, postprocess
from roi_processor import filter_after_roi_selection, extract_image_stock, combine_images, pp_one

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
        self.root.iconbitmap(current_dir + "\\images\\synaptocatcher.ico")
        
        # Load previous parameters if they exist
        self.params = self.load_params()                
        
        self.create_label_and_entry("Select File (CSZ or Excel table):", self.browse_protocol, readonly=True, attr_name="protocol")
        self.create_label_and_entry("Experiment Number:", default_value='all', attr_name="rows")
        self.create_label_and_entry("slice start:", default_value='2', attr_name="slice_start")
        self.create_label_and_entry("slice end:", default_value='6', attr_name="slice_end")
        
        self.create_separator()        
        self.create_button("1. Select ROI", self.select_roi)
        
        self.create_separator()
        self.create_label_and_entry("Filter radius:", default_value=17, attr_name="filter_radius")
        self.create_button("2. Filter", self.filter_action)
        
        self.create_separator()
        self.create_label_and_option_menu("Binarization Method:", ['otsu', 'max_entropy', 'yen', 'li', 'isodata', 'mean', 'minimum'], default_value='otsu', attr_name="binarization")
        self.create_label_and_entry("Min size of an object:", default_value=20, attr_name="min_size")
        self.create_label_and_entry("Max size of an object:", default_value=200, attr_name="max_size")
        # self.create_label_and_entry("Pixel to micron ratio:", default_value=0.1, attr_name="pixel_to_micron_ratio")
        self.create_button("3. Binarize", self.binarize_action)
        self.create_button("Remove bad spots", self.remove_ccp_action)
        
        # Elements for postprocess
        self.create_separator()
        self.create_label_and_entry("Output Directory:", self.browse_output_dir, attr_name="output_dir")        
        self.create_button("4. Combine images", self.combine_images_action)
        self.create_button("5. Postprocess (Result table)", self.run_postprocess)
        
        # Create text area for logs and redirect stdout
        self.create_text_area_with_redirect()
        
        self.previous_percentage = -1
        
        self.total_symbols = 20
        self.printed_symbols = 0
        
        self.check_file_type()
        
        self.add_hyperlink("Source: github.com/ta3map/Synapto_Catcher", "https://github.com/ta3map/Synapto_Catcher")
        
    def create_separator(self):
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=10)
    
    def create_text_area_with_redirect(self):
        self.text_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=50, height=15)
        self.text_area.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)    
        redirect_text = RedirectText(self.text_area)
        sys.stdout = redirect_text
        # button = Button(self.root, text="Print Test", command=self.on_button_click, width=20)
        # button.pack(padx=5, pady=5)
    
    def create_label_and_entry(self, label_text, command=None, default_value='', readonly=False, attr_name=None):
        frame = tk.Frame(self.root)
        frame.pack(padx=5, pady=5, fill=tk.X)
        
        label = Label(frame, text=label_text, width=30)
        label.pack(side=tk.LEFT)
        
        entry_var = StringVar(value=self.params.get(attr_name, default_value))
        entry = Entry(frame, textvariable=entry_var, width=20, state='readonly' if readonly else 'normal')
        entry.pack(side=tk.LEFT, padx=5)
        entry_var.trace_add("write", lambda *args: self.save_params(attr_name, entry_var.get()))

        if command:         
            button = Button(frame, text="...", command=command)
            button.pack(side=tk.LEFT, padx=5)
        
        if attr_name:
            setattr(self, f"{attr_name}_entry", entry)
            setattr(self, f"selected_{attr_name}", entry_var)

    def create_button(self, text, command):
        button = Button(self.root, text=text, command=command, width=20)
        button.pack(padx=5, pady=5)

    def create_label_and_option_menu(self, label_text, options, default_value, attr_name):
        frame = tk.Frame(self.root)
        frame.pack(padx=5, pady=5, fill=tk.X)
        
        label = Label(frame, text=label_text, width=30)
        label.pack(side=tk.LEFT)
        
        var = StringVar(value=self.params.get(attr_name, default_value))
        option_menu = OptionMenu(frame, var, *options)
        option_menu.pack(side=tk.LEFT, padx=5)
        var.trace_add("write", lambda *args: self.save_params(attr_name, var.get()))

        setattr(self, f"{attr_name}_method", var)
    
    def toggle_entry(self, attr_name, enabled):
        entry = getattr(self, f"{attr_name}_entry", None)
        entry_var = getattr(self, f"selected_{attr_name}", None)
        if entry and entry_var:
            if enabled:
                entry.config(state='normal')
                entry_var.set('all')
            else:
                entry.config(state='disabled')
                entry_var.set('')
    def check_file_type(self):
        attr_name = 'protocol'
        protocol_path = self.params.get(attr_name, '')
        print('Main File:')
        print(protocol_path)
        if protocol_path.endswith('.czi'):
            self.toggle_entry("rows", False)
        else:
            self.toggle_entry("rows", True) 
        
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
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            print(f"File '{file_path}' not found")
    
    def add_file_links_from_list(self, files_out):
        files_out = self.flatten_list(files_out)
        for filepath in files_out:
            self.add_file_link(filepath, filepath)
    def browse_protocol(self):
        protocol_path = filedialog.askopenfilename(filetypes=[("Protocol files", "*.czi;*.xlsx")])
        if protocol_path:
            self.selected_protocol.set(protocol_path)
            self.save_params('protocol', protocol_path)
            protocol_dir = os.path.dirname(protocol_path)
            default_output_dir = os.path.join(protocol_dir, "results")
            self.selected_output_dir.set(default_output_dir)
            self.save_params('output_dir', default_output_dir)
            
            self.check_file_type()
            
            self.prepare_data()
            

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
        if value == 0:
            print("0%------50%------100%")

    def select_roi(self):
        df, rows_to_process = self.prepare_data()
        total = len(rows_to_process)
        files_out = []
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            location = df.iloc[row_idx]['location']
            slice_start = int(self.slice_start_entry.get())
            slice_end = int(self.slice_end_entry.get())            
            files_out.append(process_file(file_path, location, slice_start, slice_end))
            self.update_progress_bar(idx, total)
        print("all ROI extracted successfully.")
        self.add_file_links_from_list(files_out)

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

    async def async_binarize_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            total = len(rows_to_process)
            files_out = []
            for idx, row_idx in enumerate(rows_to_process):
                file_path = df.iloc[row_idx]['filepath']
                row = df.iloc[row_idx]
                files_out.append(binarize_images(
                    file_path, row,
                    self.binarization_method.get(), int(self.min_size_entry.get()), 
                    int(self.max_size_entry.get())
                ))
                self.update_progress_bar(idx, total)
                await asyncio.sleep(0)  # Let's give management another task
            print("Binarization completed successfully.")
            self.add_file_links_from_list(files_out)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

    def binarize_action(self):
        asyncio.run_coroutine_threadsafe(self.async_binarize_action(), self.loop)



    def remove_ccp_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            csv_file_path = self.selected_protocol.get()
            # pixel_to_micron_ratio = float(self.pixel_to_micron_ratio_entry.get())
            remove_ccp(df, csv_file_path, rows_to_process)
            print("Binarization completed successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
    

    async def async_combine_images_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            output_directory = self.selected_output_dir.get()
            total = len(rows_to_process)
            files_out = []
            for idx, row_idx in enumerate(rows_to_process):
                file_path = df.iloc[row_idx]['filepath']
                files_out.append(combine_images(file_path, output_directory))
                self.update_progress_bar(idx, total)
                await asyncio.sleep(0)  # Let's give management another task
            print("Images combined successfully.")
            self.add_file_links_from_list(files_out)
        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()

    def combine_images_action(self):
        asyncio.run_coroutine_threadsafe(self.async_combine_images_action(), self.loop)

    async def async_run_postprocess(self):
        df, rows_to_process = self.prepare_data()
        output_directory = self.selected_output_dir.get()
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        total = len(rows_to_process)
        summary_data_list = []
    
        files_out = []
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            row = df.iloc[row_idx]
            summary_data, summary_result_path = pp_one(file_path, row, output_directory)
            files_out.append([summary_result_path])
            if summary_data is not None:
                summary_data_list.append(summary_data)
            
            self.update_progress_bar(idx, total)
            await asyncio.sleep(0)  # Let's give management another task
    
        # Create a DataFrame with collected data
        summary_df = pd.concat(summary_data_list, ignore_index=True)
        summary_df.drop(summary_df.columns[[0, -1]], axis=1, inplace=True)
        
        # Save the updated DataFrame to a new file
        summary_output_path = join(output_directory, "collected_roi_summary_data.xlsx")
        summary_df.to_excel(summary_output_path, index=False)
        
        print("Postprocessing completed successfully.")
        self.add_file_links_from_list(files_out)
        print("Final table:")
        self.add_file_links_from_list([[summary_output_path]])

    def run_postprocess(self):
        asyncio.run_coroutine_threadsafe(self.async_run_postprocess(), self.loop)

    def prepare_data(self):
        protocol_path = self.selected_protocol.get()
        required_columns = ['filepath', 'comment', 'location', 'Experiment_Number', 'take_to_stat', 'Postnatal_Age']    
        # Check if the protocol_path ends with .czi
        if protocol_path.endswith('.czi'):
            # Create a DataFrame with the specified columns and values
            df = pd.DataFrame({
                'filepath': [protocol_path],
                'comment': [''],
                'location': ['unknown'],
                'Experiment_Number': [''],
                'take_to_stat': [''],
                'Postnatal_Age': ['']
            })
            # Set rows_to_process to the first index (0)
            rows_to_process = [0]
        else:            
            experiment_numbers = self.get_exps_to_process()
            # Determine the reading method based on the file extension
            if protocol_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(protocol_path)
            else:
                df = pd.read_csv(protocol_path, delimiter=';')    
            # Check if the DataFrame contains all the required columns
            missing_columns = [column for column in required_columns if column not in df.columns]
            if missing_columns:
                messagebox.showerror("Error", f"Selected protocol file is missing required columns: {', '.join(missing_columns)}. Please choose a different file.")
                print('Error! Please choose a different file.')
                return None, None    
            # Filter out rows where 'take_to_stat' is 'no' and log removed experiments
            removed_experiments = df[df['take_to_stat'] == 'no']['Experiment_Number'].tolist()
            df = df[df['take_to_stat'] != 'no']
            print(f"Removed experiments: {removed_experiments}")    
            # Reset the indices of the DataFrame
            df.reset_index(drop=True, inplace=True)    
            # Find rows with the specified experiment numbers
            if experiment_numbers == 'all':
                rows_to_process = list(range(len(df)))
            else:
                rows_to_process = df[df['Experiment_Number'].isin(experiment_numbers)].index.tolist()
        return df, rows_to_process

    def get_exps_to_process(self):
        exps_input = self.rows_entry.get()
        if not self.selected_protocol.get() or not exps_input:
            raise ValueError("Please select a protocol and enter experiments to process.")
        return self.parse_exps(exps_input)

    def parse_exps(self, exps_input):
        exps_input = exps_input.strip().lower()
        if exps_input == 'all':
            return 'all'
        try:
            # Check if input is in the range format
            if ':' in exps_input:
                start, end = map(int, exps_input.split(':'))
                return list(range(start, end + 1))
            else:
                return [int(exp.strip()) for exp in exps_input.split(',')]
        except ValueError:
            raise ValueError("Invalid experiments input. Enter comma-separated numbers, 'all', or a range in the format 'start:end'.")

    def browse_output_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.selected_output_dir.set(directory)
            self.save_params('output_dir', directory)

    def save_params(self, key, value):
        self.params[key] = value
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
    
if __name__ == "__main__":
    root = tk.Tk()
    app = ROIAnalyzerApp(root)

    # Determine the screen dimensions
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    
    # Determine the window dimensions
    window_width = 400
    window_height = 900
    
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
    
    # Create a new asyncio event loop and run it in a separate thread
    new_loop = asyncio.new_event_loop()
    t = Thread(target=start_event_loop, args=(new_loop,))
    t.start()

    app.set_loop(new_loop)

    root.mainloop()
    new_loop.call_soon_threadsafe(new_loop.stop)  # Stop the event loop when the window closes
