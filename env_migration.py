import subprocess
import json
import os

def get_installed_modules(env_path):
    print(f"Getting the list of modules from the environment: {env_path}")
    result = subprocess.run([env_path, '-m', 'pip', 'list', '--format', 'json'], capture_output=True, text=True)
    modules = json.loads(result.stdout)
    print(f"Found {len(modules)} modules in {env_path}")
    return modules

def install_module(env_path, module, version=None):
    if version:
        print(f"Installing module {module} version {version} in the environment: {env_path}")
        subprocess.run([env_path, '-m', 'pip', 'install', f'{module}=={version}'], capture_output=True, text=True)
    else:
        print(f"Installing the latest version of module {module} in the environment: {env_path}")
        subprocess.run([env_path, '-m', 'pip', 'install', module], capture_output=True, text=True)
    print(f"Module {module} installed in {env_path}")

def check_installed_modules(modules_to_install, light_modules_dict):
    modules_to_skip = []
    for mod in modules_to_install:
        module_name = mod['name']
        version = mod['version']
        if module_name in light_modules_dict and (version is None or light_modules_dict[module_name] == version):
            modules_to_skip.append(mod)
            print(f"Module {module_name} version {version if version else 'latest version'} is already installed in the light environment, skipping installation.")
    return [mod for mod in modules_to_install if mod not in modules_to_skip]

# Paths to your environments
home_dir = os.path.expanduser('~')
classic_env = os.path.join(home_dir, 'anaconda3', 'python.exe')
light_env = os.path.join(home_dir, 'anaconda3', 'envs', 'light_env', 'python.exe')

print("Getting the list of modules from the classic environment...")
# Get the list of modules
classic_modules = get_installed_modules(classic_env)
print("Getting the list of modules from the light environment...")
light_modules = get_installed_modules(light_env)

# Create a dictionary with module versions from classic
classic_modules_dict = {mod['name']: mod['version'] for mod in classic_modules}
light_modules_dict = {mod['name']: mod['version'] for mod in light_modules}

# Save modules and their versions to a JSON file
modules_to_install = []
for mod in light_modules:
    module_name = mod['name']
    version = classic_modules_dict.get(module_name, None)
    modules_to_install.append({"name": module_name, "version": version})

# Custom modules and versions
cust_modules = ['aicspylibczi', 'opencv-python', 'seaborn', 'scikit_posthocs', 'keyboard', 'ttkthemes']
cust_versions = ['', '', '', '', '', '']

# Add custom modules to the installation list
for module, version in zip(cust_modules, cust_versions):
    if module in classic_modules_dict:
        version = classic_modules_dict[module]
        print(f"Using version {version} for custom module {module} from the classic environment.")
    modules_to_install.append({"name": module, "version": version if version else None})

# Remove modules that already have the required version in the light_env
modules_to_install = check_installed_modules(modules_to_install, light_modules_dict)

json_file_path = os.path.join(os.path.dirname(__file__), 'modules_to_install.json')
with open(json_file_path, 'w') as json_file:
    json.dump(modules_to_install, json_file, indent=4)

print(f"Modules and versions saved to the file: {json_file_path}")

# Read data from the JSON file before installation
with open(json_file_path, 'r') as json_file:
    modules_to_install = json.load(json_file)

print("Starting the installation of modules in the light environment...")
# Install the required versions of modules in light_env
total_modules = len(modules_to_install)
installed_modules = 0
for mod in modules_to_install:
    module_name = mod['name']
    version = mod['version']
    install_module(light_env, module_name, version)
    installed_modules += 1
    print(f"Progress: {installed_modules}/{total_modules} modules installed.")

print("Installation process completed.")
wait = input("Press Enter to continue.")