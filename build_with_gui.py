import subprocess
import os
import shutil

# Path to your Python interpreter (assuming Anaconda and a specific environment are used)
python_interpreter = os.path.expanduser('~\\anaconda3\\envs\\light_env\\python.exe')

# Paths to files and directories (using home directory)
home_dir = os.path.expanduser('~')
config_path = os.path.join(home_dir, 'Documents', 'synapto_catcher', 'converter_settings.json')
output_dir = os.path.join(home_dir, 'Documents', 'synapto_catcher', 'output')
script_path = os.path.join(home_dir, 'Documents', 'synapto_catcher', 'syn_catch_GUI.py')

# Determine the output file name based on the input script name
script_name = os.path.splitext(os.path.basename(script_path))[0]
script_name = 'Synapto Catcher'
output_file = os.path.join(output_dir, f'{script_name}.exe')
output_dir_without_ext = os.path.join(output_dir, script_name)

# Remove the output file if it exists
if os.path.exists(output_file):
    os.remove(output_file)
    print(f'Existing file removed: {output_file}')

# Remove the output directory if it exists
if os.path.exists(output_dir_without_ext):
    shutil.rmtree(output_dir_without_ext)
    print(f'Existing directory removed: {output_dir_without_ext}')

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
