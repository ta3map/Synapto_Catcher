# %%
import json
import os
import tempfile

TEMP_FILE = os.path.join(tempfile.gettempdir(), 'synapto_catch_params.json')

# Function to load parameters
def load_params():
    if os.path.exists(TEMP_FILE):
        try:
            with open(TEMP_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            os.remove(TEMP_FILE)
    return {}

# Function to write data to file via ADS
def write_data_to_file(file_path, data):
    ads_path = f"{file_path}:syn_catch_metadata"
    with open(ads_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

# Usage example
params = load_params()
file_path = r"C:\Users\ta3ma\Pictures\snap.PNG"

write_data_to_file(file_path, params)
# %%
import json
import os


def read_metadata_from_file(file_path):
    ads_path = f"{file_path}:syn_catch_metadata"
    if os.path.exists(ads_path):
        with open(ads_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Usage example

file_path = r"E:\iMAGES\Sp P11.5 3\Experiment-05_results\Experiment-05_0_denoised.29627e8e-6d021339-4f5cf7df-33f2a515-5e40a3c5-efc46cb2.png"

retrieved_params = read_metadata_from_file(file_path)
print(retrieved_params)
