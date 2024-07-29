import subprocess
import sys
import os

# Path to your Python interpreter
python_interpreter = 'C:\\Users\\ta3ma\\anaconda3\\python.exe'
python_interpreter = 'C:\\Users\\ta3ma\\anaconda3\\envs\\light_env\\python.exe'

# Paths to files and directories
config_path = 'C:\\Users\\ta3ma\\Documents\\synapto_catcher\\converter_settings.json'
output_dir = 'C:\\Users\\ta3ma\\Documents\\synapto_catcher\\output'
script_path = 'C:\\Users\\ta3ma\\Documents\\synapto_catcher\\roi_analyzer_app.py'

# Determine the output file name based on the input script name
script_name = os.path.splitext(os.path.basename(script_path))[0]
script_name = 'Synapto Catcher'
output_file = os.path.join(output_dir, f'{script_name}.exe')

# Remove the output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f'Existing file removed: {output_file}')

# Command to run auto-py-to-exe
command = [
    python_interpreter,
    '-m', 'auto_py_to_exe',
    '--config', config_path,
    '--output-dir', output_dir,
    script_path
]

# Run the command and display the output in real time
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# Display the output in real time
for line in process.stdout:
    print(line, end='')

# Wait for the process to complete and get the return code
return_code = process.wait()

if return_code != 0:
    print(f"Process completed with an error, return code: {return_code}")
else:
    print("Process completed successfully")
