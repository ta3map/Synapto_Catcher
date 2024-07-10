from tkinter import filedialog, messagebox, Button, Label, Entry, StringVar, OptionMenu, Tk, ttk
import sys
import os
import json
import tempfile
import pandas as pd
import time
from os.path import splitext, basename, dirname, join, exists
from tqdm import tqdm

%matplotlib qt

# Adding current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from roi_processor import process_file, binarize_images, remove_ccp, postprocess
from roi_processor import filter_after_roi_selection, extract_image_stock, combine_images, pp_one

TEMP_FILE = os.path.join(tempfile.gettempdir(), 'synapto_catch_params.json')

class ROIAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI Analyzer")
        
        # Load previous parameters if they exist
        self.params = self.load_params()
        
        self.create_label_and_entry("Select Protocol:", 0, self.browse_protocol, readonly=True, attr_name="protocol")
        self.create_label_and_entry("Experiment Number:", 2, default_value='all', attr_name="rows")
        self.create_label_and_entry("slice start:", 3, default_value='2', attr_name="slice_start")
        self.create_label_and_entry("slice end:", 4, default_value='6', attr_name="slice_end")
        
        self.create_button("Extract channel", self.extract_stt, 5)
        
        self.create_button("Select ROI", self.process, 6)
        self.create_label_and_entry("Filter radius:", 7, default_value=17, attr_name="filter_radius")
        self.create_button("Filter", self.filter_action, 8)
        
        self.create_label_and_option_menu("Binarization Method:", 9, ['otsu', 'max_entropy', 'yen', 'li', 'isodata', 'mean', 'minimum'], default_value='otsu', attr_name="binarization")
        self.create_label_and_entry("Min size of an object:", 10, default_value=20, attr_name="min_size")
        self.create_label_and_entry("Max size of an object:", 11, default_value=200, attr_name="max_size")
        self.create_label_and_entry("Pixel to micron ratio:", 12, default_value=0.1, attr_name="pixel_to_micron_ratio")

        self.create_button("Binarize", self.binarize_action, 13)
        self.create_button("Remove bad spots", self.remove_ccp_action, 14)
        
        # Elements for postprocess
        self.create_label_and_entry("Output Directory:", 15, self.browse_output_dir, attr_name="output_dir")        
        self.create_button("Combine images", self.combine_images_action, 16)
        self.create_button("Run Postprocess", self.run_postprocess, 17)
        
        self.progress = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
        self.progress.grid(row=18, columnspan=2, pady=10)

    def create_label_and_entry(self, label_text, row, command=None, default_value='', readonly=False, attr_name=None):
        label = Label(self.root, text=label_text, width=30)
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        entry_var = StringVar(value=self.params.get(attr_name, default_value))
        entry = Entry(self.root, textvariable=entry_var, width=20, state='readonly' if readonly else 'normal')
        entry.grid(row=row, column=1, padx=5, pady=5)
        entry_var.trace_add("write", lambda *args: self.save_params(attr_name, entry_var.get()))

        if command:
            button = Button(self.root, text="Browse", command=command, width=20)
            button.grid(row=row + 1, column=1, padx=10, pady=5)

        if attr_name:
            setattr(self, f"{attr_name}_entry", entry)
            setattr(self, f"selected_{attr_name}", entry_var)

    def create_button(self, text, command, row):
        button = Button(self.root, text=text, command=command, width=20)
        button.grid(row=row, column=1, padx=5, pady=20)

    def create_label_and_option_menu(self, label_text, row, options, default_value, attr_name):
        label = Label(self.root, text=label_text, width=30)
        label.grid(row=row, column=0, padx=5, pady=5, sticky="w")

        var = StringVar(value=self.params.get(attr_name, default_value))
        option_menu = OptionMenu(self.root, var, *options)
        option_menu.grid(row=row, column=1, padx=5, pady=5)
        var.trace_add("write", lambda *args: self.save_params(attr_name, var.get()))

        setattr(self, f"{attr_name}_method", var)
        
    def update_progress_bar(self, value, maximum):
        self.progress['value'] = value+1
        self.progress['maximum'] = maximum
        time.sleep(0.2)
        self.root.update_idletasks()
        
    def browse_protocol(self):
        protocol_path = filedialog.askopenfilename(filetypes=[("Protocol files", "*.csv;*.xlsx;*.xls")])
        if protocol_path:
            self.selected_protocol.set(protocol_path)
            self.save_params('protocol', protocol_path)
            protocol_dir = os.path.dirname(protocol_path)
            default_output_dir = os.path.join(protocol_dir, "results")
            self.selected_output_dir.set(default_output_dir)
            self.save_params('output_dir', default_output_dir)

    def process(self):
        df, rows_to_process = self.prepare_data()
        total = len(rows_to_process)
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            location = df.iloc[row_idx]['location']
            slice_start = int(self.slice_start_entry.get())
            slice_end = int(self.slice_end_entry.get())            
            process_file(file_path, location, slice_start, slice_end)
            self.update_progress_bar(idx, total)

    def extract_stt(self):
        df, rows_to_process = self.prepare_data()
        total = len(rows_to_process)
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            location = df.iloc[row_idx]['location']
            slice_start = int(self.slice_start_entry.get())
            slice_end = int(self.slice_end_entry.get())            
            extract_image_stock(file_path, location, slice_start, slice_end)
            self.update_progress_bar(idx, total)

    def filter_action(self):
        df, rows_to_process = self.prepare_data()
        filter_radius = int(self.filter_radius_entry.get())
        total = len(rows_to_process)
        for idx, row_idx in enumerate(rows_to_process):
            file_path = df.iloc[row_idx]['filepath']
            location = df.iloc[row_idx]['location']   
            filter_after_roi_selection(filter_radius, file_path, location)
            self.update_progress_bar(idx, total)


    def binarize_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            binarize_images(
                df, self.selected_protocol.get(), rows_to_process,
                self.binarization_method.get(), int(self.min_size_entry.get()), 
                int(self.max_size_entry.get()),
                float(self.pixel_to_micron_ratio_entry.get())
            )
            messagebox.showinfo("Success", "Binarization completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            
    def remove_ccp_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            csv_file_path = self.selected_protocol.get()
            pixel_to_micron_ratio = float(self.pixel_to_micron_ratio_entry.get())
            remove_ccp(df, csv_file_path, rows_to_process, pixel_to_micron_ratio)
            messagebox.showinfo("Success", "Binarization completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            
    def combine_images_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            output_directory = self.selected_output_dir.get()
            total = len(rows_to_process)
            for idx, row_idx in enumerate(rows_to_process):
                file_path = df.iloc[row_idx]['filepath']
                combine_images(file_path, output_directory)
                self.update_progress_bar(idx, total)
                
            messagebox.showinfo("Success", "Images combined successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
                


    def run_postprocess(self):
        df, rows_to_process = self.prepare_data()
        output_directory = self.selected_output_dir.get()
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        total = len(rows_to_process)
        summary_data_list = []
    
        # Using tqdm to display the progress bar
        for idx, row_idx in tqdm(enumerate(rows_to_process), total=total, desc="Processing"):
            file_path = df.iloc[row_idx]['filepath']
            row = df.iloc[row_idx]
            summary_data = pp_one(file_path, row, output_directory)
            
            if summary_data is not None:
                summary_data_list.append(summary_data)
            
            self.update_progress_bar(idx, total)
            
        # Create a DataFrame with collected data
        summary_df = pd.concat(summary_data_list, ignore_index=True)
        summary_df.drop(summary_df.columns[[0, -1]], axis=1, inplace=True)
        
        # Save the updated DataFrame to a new file
        summary_output_path = join(output_directory, "collected_roi_summary_data.xlsx")
        summary_df.to_excel(summary_output_path, index=False)
        
        messagebox.showinfo("Success", "Postprocessing completed successfully.")


    def prepare_data(self):
        protocol_path = self.selected_protocol.get()
        experiment_numbers = self.get_exps_to_process()
    
        # Determine the reading method based on the file extension
        if protocol_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(protocol_path)
        else:
            df = pd.read_csv(protocol_path, delimiter=';')
            
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

if __name__ == "__main__":
    root = Tk()
    app = ROIAnalyzerApp(root)
    root.mainloop()
