import subprocess
import json
import os

def get_installed_modules(env_path):
    print(f"Получаем список модулей из окружения: {env_path}")
    result = subprocess.run([env_path, '-m', 'pip', 'list', '--format', 'json'], capture_output=True, text=True)
    modules = json.loads(result.stdout)
    print(f"Найдено {len(modules)} модулей в {env_path}")
    return modules

def install_module(env_path, module, version=None):
    if version:
        print(f"Устанавливаем модуль {module} версии {version} в окружении: {env_path}")
        subprocess.run([env_path, '-m', 'pip', 'install', f'{module}=={version}'], capture_output=True, text=True)
    else:
        print(f"Устанавливаем последнюю версию модуля {module} в окружении: {env_path}")
        subprocess.run([env_path, '-m', 'pip', 'install', module], capture_output=True, text=True)
    print(f"Модуль {module} установлен в {env_path}")

def check_installed_modules(modules_to_install, light_modules_dict):
    modules_to_skip = []
    for mod in modules_to_install:
        module_name = mod['name']
        version = mod['version']
        if module_name in light_modules_dict and (version is None or light_modules_dict[module_name] == version):
            modules_to_skip.append(mod)
            print(f"Модуль {module_name} версии {version if version else 'последней версии'} уже установлен в light окружении, пропускаем установку.")
    return [mod for mod in modules_to_install if mod not in modules_to_skip]

# Пути к вашим окружениям
classic_env = "C:\\Users\\ta3ma\\anaconda3\\python.exe"
light_env = "C:\\Users\\ta3ma\\anaconda3\\envs\\light_env\\python.exe"

print("Получение списка модулей из classic окружения...")
# Получаем список модулей
classic_modules = get_installed_modules(classic_env)
print("Получение списка модулей из light окружения...")
light_modules = get_installed_modules(light_env)

# Создаем словарь с версиями модулей из classic
classic_modules_dict = {mod['name']: mod['version'] for mod in classic_modules}
light_modules_dict = {mod['name']: mod['version'] for mod in light_modules}

# Сохраняем модули и их версии в JSON файл
modules_to_install = []
for mod in light_modules:
    module_name = mod['name']
    version = classic_modules_dict.get(module_name, None)
    modules_to_install.append({"name": module_name, "version": version})

# Кастомные модули и версии
cust_modules = ['aicspylibczi', 'opencv-python']
cust_versions = ['', '']

# Добавляем кастомные модули к списку для установки
for module, version in zip(cust_modules, cust_versions):
    if module in classic_modules_dict:
        version = classic_modules_dict[module]
        print(f"Используем версию {version} для кастомного модуля {module} из classic окружения.")
    modules_to_install.append({"name": module, "version": version if version else None})

# Убираем модули, которые уже имеют нужную версию в light_env
modules_to_install = check_installed_modules(modules_to_install, light_modules_dict)

json_file_path = os.path.join(os.path.dirname(__file__), 'modules_to_install.json')
with open(json_file_path, 'w') as json_file:
    json.dump(modules_to_install, json_file, indent=4)

print(f"Модули и версии сохранены в файл: {json_file_path}")

# Читаем данные из JSON файла перед установкой
with open(json_file_path, 'r') as json_file:
    modules_to_install = json.load(json_file)

print("Начинаем установку модулей в light окружении...")
# Устанавливаем необходимые версии модулей в light_env
total_modules = len(modules_to_install)
installed_modules = 0
for mod in modules_to_install:
    module_name = mod['name']
    version = mod['version']
    install_module(light_env, module_name, version)
    installed_modules += 1
    print(f"Прогресс: {installed_modules}/{total_modules} модулей установлено.")

print("Процесс установки завершен.")
