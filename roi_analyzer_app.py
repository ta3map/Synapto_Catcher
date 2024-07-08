from tkinter import filedialog, messagebox, Button, Label, Entry, StringVar, OptionMenu, Tk
from pandas import read_csv
import sys
import os
import json
import tempfile

# Adding current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from roi_processor import process_file, binarize_images, remove_ccp, postprocess

TEMP_FILE = os.path.join(tempfile.gettempdir(), 'synapto_catch_params.json')

class ROIAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI Analyzer")

        # Load previous parameters if they exist
        self.params = self.load_params()

        self.create_label_and_entry("Select Protocol:", 0, self.browse_protocol, readonly=True, attr_name="protocol")
        self.create_label_and_entry("Rows to Process:", 3, default_value='all', attr_name="rows")

        self.create_button("Select ROI", self.process, 4)
        self.create_label_and_option_menu("Binarization Method:", 5, ['otsu', 'max_entropy', 'yen', 'li', 'isodata', 'mean', 'minimum'], default_value='otsu', attr_name="binarization")
        self.create_label_and_entry("Min size of an object:", 6, default_value=50, attr_name="min_size")
        self.create_label_and_entry("Pixel to micron ratio:", 7, default_value=0.1, attr_name="pixel_to_micron_ratio")

        self.create_button("Binarize", self.binarize_action, 8)
        self.create_button("Remove bad spots", self.remove_ccp_action, 9)
        
        # Elements for postprocess
        self.create_label_and_entry("Output Directory:", 11, self.browse_output_dir, attr_name="output_dir")
        self.create_button("Run Postprocess", self.run_postprocess, 13)

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

    def browse_protocol(self):
        protocol_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if protocol_path:
            self.selected_protocol.set(protocol_path)
            self.save_params('protocol', protocol_path)
            protocol_dir = os.path.dirname(protocol_path)
            default_output_dir = os.path.join(protocol_dir, "results")
            self.selected_output_dir.set(default_output_dir)
            self.save_params('output_dir', default_output_dir)

    def process(self):
        try:
            df, rows_to_process = self.prepare_data()
            for idx in rows_to_process:
                file_path = df.iloc[idx]['filepath']
                location = df.iloc[idx]['location']
                slice_start = int(df.iloc[idx]['slice_start'])
                slice_end = int(df.iloc[idx]['slice_end'])
                process_file(file_path, location, slice_start, slice_end)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")


    def binarize_action(self):
        try:
            df, rows_to_process = self.prepare_data()
            binarize_images(
                df, self.selected_protocol.get(), rows_to_process,
                self.binarization_method.get(), int(self.min_size_entry.get()), 
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
            
            
    def run_postprocess(self):
        try:
            df, rows_to_process = self.prepare_data()
            postprocess(df, self.selected_protocol.get(), self.selected_output_dir.get(), rows_to_process)
            
            if len(rows_to_process) > 1:
                messagebox.showinfo("Success", "Postprocessing completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def prepare_data(self):
        protocol_path = self.selected_protocol.get()
        rows_to_process = self.get_rows_to_process()

        df = read_csv(protocol_path, delimiter=';')
        if rows_to_process == 'all':
            rows_to_process = list(range(len(df)))
        else:
            rows_to_process = [row - 2 for row in rows_to_process]
            rows_to_process = [row for row in rows_to_process if row > -1]# remove all negative
            
        return df, rows_to_process

    def get_rows_to_process(self):
        rows_input = self.rows_entry.get()
        if not self.selected_protocol.get() or not rows_input:
            raise ValueError("Please select a protocol and enter rows to process.")
        
        return self.parse_rows(rows_input)

    def parse_rows(self, rows_input):
        if rows_input.strip().lower() == 'all':
            return 'all'
        try:
            return [int(row.strip()) for row in rows_input.split(',')]
        except ValueError:
            raise ValueError("Invalid rows input. Enter comma-separated numbers or 'all'.")

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
