%matplotlib qt
#%%
import subprocess

# Путь к исполняемому файлу ImageJ
imagej_path = r"C:\Program Files (x86)\Fiji.app\ImageJ-win32.exe"
# Путь к вашему макросу
macro_path = r"E:\iMAGES\Macro_synapto_avec_protocol_no_dialog.ijm"

# Запуск макроса
subprocess.run([imagej_path, "--run", macro_path])
#%% постпроцессинг и модификация таблицы с данными

import pandas as pd
import os
import re
from PIL import Image

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"
output_directory = "E:\\iMAGES\\processed_images"

# Создаем выходную директорию, если ее нет
os.makedirs(output_directory, exist_ok=True)

# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')
df = df[df['take_to_stat'] != 'no']

# Функция для объединения изображений и сохранения их
def combine_and_save_images(image1_path, image2_path, output_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Определяем размеры нового изображения
    combined_width = image1.width + image2.width
    combined_height = max(image1.height, image2.height)
    
    # Создаем новое изображение
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    
    # Сохраняем объединенное изображение
    combined_image.save(output_path)

# Функция для парсинга данных из строки
def parse_slice_info(slice_info):
    pattern = r"P(\d+)(?:\.\d+)*\s+slide(\d+)\s+slice(\d+)_Experiment-(\d+)"
    match = re.search(pattern, slice_info)
    if match:
        postnatal_age = int(match.group(1))  # берем только целое число
        slide_number = int(match.group(2))
        slice_number = int(match.group(3))
        experiment_number = int(match.group(4))
        return postnatal_age, slide_number, slice_number, experiment_number
    return None, None, None, None

# Коллекционируем данные и объединяем изображения
summary_data_list = []

for index, row in df.iterrows():
    image_path = row['filepath']
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    experiment_date = os.path.basename(os.path.dirname(image_path))
    
    # Пути к изображениям
    denoised_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
    masks_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_masks_Zprojection_crop.tif")
    
    # Объединяем и сохраняем изображения
    combined_image_path = os.path.join(output_directory, f"{experiment_date}_{base_name}_combined.tif")
    try:
        combine_and_save_images(denoised_image_path, masks_image_path, combined_image_path)
    except Exception as e:
        print(f"Error combining images for {image_path}: {e}")
        continue
    
    # Путь к файлу с результатами
    summary_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_summary_result_table.xls")
    
    # Сохраняем первую строку таблицы результатов
    summary_data = pd.read_csv(summary_result_path, sep='\t')
    if summary_data is not None:
        summary_data['filepath'] = image_path
        summary_data['location'] = row['location']
        summary_data['threshold_method'] = row['threshold_method']
        summary_data_list.append(summary_data)

# Создаем DataFrame с коллекционированными данными
summary_df = pd.concat(summary_data_list, ignore_index=True)
summary_df.drop(summary_df.columns[[0, -1]], axis=1, inplace=True)

# Добавление новых колонок в DataFrame
summary_df['Postnatal_Age'] = None
summary_df['Slide_Number'] = None
summary_df['Slice_Number'] = None
summary_df['Experiment_Number'] = None

# Парсинг данных и заполнение новых колонок
for index, row in summary_df.iterrows():
    slice_info = row['Slice']
    postnatal_age, slide_number, slice_number, experiment_number = parse_slice_info(slice_info)
    summary_df.at[index, 'Postnatal_Age'] = postnatal_age
    summary_df.at[index, 'Slide_Number'] = slide_number
    summary_df.at[index, 'Slice_Number'] = slice_number
    summary_df.at[index, 'Experiment_Number'] = experiment_number

# Сохранение обновленного DataFrame в новый файл
summary_output_path = os.path.join(output_directory, "collected_summary_data.xlsx")
summary_df.to_excel(summary_output_path, index=False)

print("Постпроцессинг завершен. Обновленные данные сохранены в:", summary_output_path)

#%% денойзинг
%matplotlib qt
import pandas as pd
import czifile
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift
from skimage import exposure
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"

# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')

idx = 164

slice_start = int(df.iloc[idx]['slice_start'])
slice_end = int(df.iloc[idx]['slice_end'])

file_path = df.iloc[idx]['filepath']

czi = czifile.CziFile(file_path)

image_data = czi.asarray()

slide = list(range(slice_start, slice_end + 1))

channel_1 = 0  # Убедитесь, что правильно указали канал
sample_slice_1 = np.mean(image_data[0, 0, channel_1, slide, :, :, 0], axis=0)

# Шаг 1: Частотная фильтрация
def frequency_filtering(image):
    # Преобразование Фурье
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    
    # Создание маски для фильтрации низких частот
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 5  # Радиус области, которую нужно удалить
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
    mask[mask_area] = 0
    
    # Применение маски и обратное преобразование Фурье
    f_transform_shifted_filtered = f_transform_shifted * mask
    f_transform_filtered = fftshift(f_transform_shifted_filtered)
    image_filtered = np.abs(ifft2(f_transform_filtered))
    
    return image_filtered

# Шаг 2: Повышение контраста
def enhance_contrast(image):
    # Используем equalize_adapthist для повышения контраста
    image_enhanced = exposure.equalize_adapthist(image / np.max(image))
    return image_enhanced

# Шаг 3: Бинаризация методом Otsu
def binarize_image(image):
    # Применение метода Otsu для автоматической бинаризации
    threshold_value = get_threshold_value(image, 'otsu')
    binary = image > threshold_value
    return binary

def get_threshold_value(image_array, binarization_method):
    if binarization_method == 'max_entropy':
        threshold_value = max_entropy_threshold(image_array)
    elif binarization_method == 'otsu':
        threshold_value = threshold_otsu(image_array)
    elif binarization_method == 'yen':
        threshold_value = threshold_yen(image_array)
    elif binarization_method == 'li':
        threshold_value = threshold_li(image_array)
    elif binarization_method == 'isodata':
        threshold_value = threshold_isodata(image_array)
    elif binarization_method == 'mean':
        threshold_value = threshold_mean(image_array)
    elif binarization_method == 'minimum':
        threshold_value = threshold_minimum(image_array)
    else:
        raise ValueError(f"Unsupported binarization method: {binarization_method}")
    
    return threshold_value

def max_entropy_threshold(image):
    hist, bin_edges = histogram(image.ravel(), bins=256, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    entropy = -hist * log(hist + finfo(float).eps)
    threshold = bin_mids[argmax(entropy)]
    return threshold


# Применяем частотную фильтрацию
filtered_image = frequency_filtering(sample_slice_1)

# Применяем повышение контраста
# enhanced_image = enhance_contrast(filtered_image)

# Применяем бинаризацию методом Otsu
binary_image = binarize_image(filtered_image)
binary_image = remove_small_objects(binary_image, min_size=50)

# Отображаем результаты
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(sample_slice_1, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Frequency Filtering')
plt.imshow(filtered_image, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Otsu Binarization')
plt.imshow(binary_image, cmap='gray')

plt.show()


#%% Бинаризация после денойзинга (без ROI)
import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from skimage import filters, measure, morphology

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"
rows_to_process = [0]  # Укажите номера строк для обработки (начиная с 0 для первой строки)
pixel_to_micron_ratio = 0.1  # Коэффициент перевода из пикселей в микрометры (предположение)

# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')

for row_number_to_process in rows_to_process:
    # Получаем строку для обработки
    row = df.iloc[row_number_to_process]
    
    # Проверяем значение take_to_stat
    if row['take_to_stat'] == 'no':
        print(f"Skipping row {row_number_to_process} because take_to_stat is 'no'")
    else:
        image_path = row['filepath']
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        experiment_date = os.path.basename(os.path.dirname(image_path))
    
        # Пути к изображениям
        denoised_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
        masks_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_masks_Zprojection_crop.tif")
        full_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_full_result_table.xls")
        summary_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_summary_result_table.xls")
    
        # Чтение денойзированного изображения
        image = Image.open(denoised_image_path).convert('L')
        
        image_array = np.array(image)
    
        # Бинаризация изображения
        threshold_value = filters.threshold_otsu(image_array)
        binary_image = image_array > threshold_value
        binary_image = morphology.remove_small_objects(binary_image, min_size=64)
    
        # Сохранение бинаризованного изображения
        binary_image_pil = Image.fromarray((binary_image * 255).astype(np.uint8))
        binary_image_pil = ImageOps.invert(binary_image_pil)  # Инвертируем изображение
        binary_image_pil.save(masks_image_path)
    
        # Поиск объектов и измерение параметров
        labeled_image = measure.label(binary_image)
        props = measure.regionprops(labeled_image, intensity_image=image_array)
    
        # Подготовка данных для сохранения
        results = []
        total_objects = 0
        total_area = 0
        total_mean_intensity = 0
        for index, prop in enumerate(props, start=1):
            area_microns = prop.area * pixel_to_micron_ratio**2
            total_objects += 1
            total_area += area_microns
            total_mean_intensity += prop.mean_intensity * area_microns
            results.append({
                "": index,  # Индексы объектов
                "Area": f"{area_microns:.3f}",
                "Mean": f"{prop.mean_intensity:.3f}",
                "Min": int(prop.min_intensity),
                "Max": int(prop.max_intensity)
            })
    
        # Сохранение данных до суммирования
        results_df = pd.DataFrame(results)
        results_df.to_csv(full_result_path, sep='\t', index=False)
    
        # Подготовка суммарных данных
        if total_objects > 0:
            average_size = total_area / total_objects
            average_mean_intensity = total_mean_intensity / total_area
        else:
            average_size = 0
            average_mean_intensity = 0
    
        summary_result = {
            "": 1,  # Пустая колонка с единицей в первой строке
            "Slice": os.path.basename(masks_image_path),
            "Count": total_objects,
            "Total Area": f"{total_area:.3f}",
            "Average Size": f"{average_size:.3f}",
            "%Area": f"{(total_area / (binary_image.size * pixel_to_micron_ratio**2)) * 100:.3f}",
            "Mean": f"{average_mean_intensity:.3f}"
        }
        
        # Сохранение суммарных данных
        summary_df = pd.DataFrame([summary_result])
        summary_df.to_csv(summary_result_path, sep='\t', index=False)
    
        print(f"Row {row_number_to_process} processed. Saved binary image as {masks_image_path}.")

#%% статистика с виолинами (без ROI)


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import numpy as np

# Load the data
file_path = 'E:/iMAGES/processed_images_full/collected_summary_data.xlsx'
data = pd.read_excel(file_path)

# Plotting function
def plot_numerical_parameters(data, parameters, category, hue):
    for parameter in parameters:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=category, y=parameter, hue=hue, data=data, palette="deep")
        plt.title(f'Relationship between {parameter} and {category} by {hue}')
        plt.xlabel(category)
        plt.ylabel(parameter)
        plt.legend(title=hue)
        plt.show()

# List of numerical parameters
numerical_parameters = ['Count', 'Total Area', 'Average Size', '%Area', 'Mean']

# Plot the data with respect to Postnatal_Age and color-coded by location
plot_numerical_parameters(data, numerical_parameters, 'Postnatal_Age', 'location')

# Filtering the data for Postnatal_Age 5, 11, and 15
filtered_data = data[data['Postnatal_Age'].isin([5, 11, 15])]

# Violin plot function
def plot_violin(data, parameter, category, hue):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=category, y=parameter, hue=hue, data=data, split=True, palette="deep")
    plt.title(f'Violin plot of {parameter} by {category} and {hue}')
    plt.xlabel(category)
    plt.ylabel(parameter)
    plt.legend(title=hue)
    plt.show()

# Plotting violin plots for the numerical parameters
for parameter in numerical_parameters:
    plot_violin(filtered_data, parameter, 'Postnatal_Age', 'location')

# Performing statistical comparisons
stat_results = []
for parameter in numerical_parameters:
    age_groups = [filtered_data[filtered_data['Postnatal_Age'] == age][parameter] for age in [5, 11, 15]]
    stat, p = kruskal(*age_groups)
    stat_results.append({'Parameter': parameter, 'Kruskal-Wallis H': stat, 'p-value': p})

# Creating a dataframe for the statistical results
stat_results_df = pd.DataFrame(stat_results)

# Function to calculate pairwise p-values using Mann-Whitney U test
def pairwise_comparisons(data, parameter, groups, group_col='Postnatal_Age'):
    p_values = np.zeros((len(groups), len(groups)))
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:
                group1_data = data[data[group_col] == group1][parameter]
                group2_data = data[data[group_col] == group2][parameter]
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                p_values[i, j] = p_value
                p_values[j, i] = p_value
            elif i == j:
                p_values[i, j] = np.nan
    
    p_values_df = pd.DataFrame(p_values, index=groups, columns=groups)
    return p_values_df

# List of groups to compare
groups = [5, 11, 15]

# Initialize a dictionary to store pairwise p-values for each parameter
pairwise_p_values = {}

# Calculate pairwise comparisons for each parameter
for parameter in numerical_parameters:
    pairwise_p_values[parameter] = pairwise_comparisons(filtered_data, parameter, groups)

# Display the pairwise p-values
for parameter, p_values_df in pairwise_p_values.items():
    print(f"Pairwise p-values for {parameter}:")
    print(p_values_df)


#%% 1 Определяем ROI для анализа

%matplotlib qt

import czifile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path
from matplotlib.patches import Polygon

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"


# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')
# df = df[df['take_to_stat'] != 'no']
# df = df[df['location'] == 'SO']

rows_to_process = list(range(0,len(df)))  # Укажите номера строк для обработки или 'all' для анализа всех строк

# по списку номеров экспериментов
exp_list = [512, 513, 516, 517, 524, 525, 528, 529, 533, 534, 537, 538, 541, 542, 665]  # Пример списка номеров экспериментов
rows_to_process = np.asarray(df['Experiment_Number'].isin(exp_list)).nonzero()[0].tolist()

rows_to_process = [1]

if rows_to_process == 'all':
    rows_to_process = range(len(df))
    
# Функция для обработки каждого файла
def process_file(file_path, location):
    with czifile.CziFile(file_path) as czi:
        # Получение данных изображения
        image_data = czi.asarray()
        print("Image shape:", image_data.shape)
        
        # Пример доступа к данным изображения
        # В зависимости от формата файла может потребоваться изменить индексы
        channel_1 = 0
        channel_3 = 3
        
        if np.size(image_data,2) == 3:
            channel_3 = 2
            print("3 channels instead of 4")
        
        slide = 2
        slice_start = 2
        slice_end = 6
        
        slide = list(range(slice_start, slice_end+1))
        # Извлечение данных для каждого канала
        sample_slice_1 = np.mean(image_data[0, 0, channel_1, slide, :, :, 0], axis = 0)        
        sample_slice_3 = np.mean(image_data[0, 0, channel_3, slide, :, :, 0], axis = 0)
        
        # Объединение изображений в одно с различными цветами
        # Создание пустого RGB изображения
        combined_image = np.zeros((*sample_slice_1.shape, 3), dtype=np.uint8)
        
        # Нормализация изображений для лучшего отображения
        sample_slice_1_normalized = (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1)) * 255
        sample_slice_3_normalized = (sample_slice_3 - np.min(sample_slice_3)) / (np.max(sample_slice_3) - np.min(sample_slice_3)) * 255
        
        # Назначение цветов
        combined_image[:, :, 0] = sample_slice_1_normalized  # Красный канал
        combined_image[:, :, 2] = sample_slice_3_normalized  # Синий канал
        
    # Определение разрешения из combined_image
    height, width = combined_image.shape[:2]
    dpi = 200
    figsize = width / float(dpi), height / float(dpi)
    
    # Визуализация объединенного изображения с возможностью интерактивного выбора области
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi*0.8)  # Уменьшение размера окна
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы
    ax.imshow(combined_image)
    # ax.set_title(f"{file_path} - {location}", fontsize=5)
    ax.axis('off')  # Убираем отображение осей
    # Добавление текста на изображение
    ax.text(0.5, 0.95, f"{file_path} - {location}", transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))

    # Интерактивный выбор области
    coords = []

    def onselect(verts):
        coords.extend(verts)
        polygon = Polygon(verts, closed=True, edgecolor='#1DE720', facecolor='none', linewidth=2, alpha=0.7)
        ax.add_patch(polygon)
        plt.draw()
        plt.close(fig)

    lineprops = dict(color='#1DE720', linestyle='-', linewidth=2, alpha=0.7)
    polygon_selector = PolygonSelector(ax, onselect, props=lineprops)

    # Ждем завершения интерактивного выбора области
    plt.show(block=True)

    # Сохранение координат в файл
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    experiment_date = os.path.basename(os.path.dirname(file_path))
    
    coords_file_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_roi_coords.csv")
    
    coords_df = pd.DataFrame(coords, columns=['x', 'y'])
    coords_df.to_csv(coords_file_path, sep=';', index=False)
    print(f"Coordinates saved to {coords_file_path}")

    # Сохранение изображения с нарисованным ROI
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы
    ax.imshow(combined_image)
    polygon = Polygon(coords, closed=True, edgecolor='#1DE720', facecolor='none', linewidth=2, alpha=0.7)
    ax.add_patch(polygon)
    
    # Добавление текста на изображение
    ax.text(0.5, 0.95, f"{file_path} - {location}", transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))
    
    ax.axis('off')
    
    image_file_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_with_roi.png")
    plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(fig)
    print(f"Image with ROI saved to {image_file_path}")

# Обработка указанных строк
for idx in rows_to_process:
    file_path = df.iloc[idx]['filepath']
    location = df.iloc[idx]['location']
    process_file(file_path, location)


#%% ROI с размером во всю картинку для SR

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"


# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')
df = df[df['take_to_stat'] != 'no']
df = df[df['location'] == 'SP']



rows_to_process = list(range(0, len(df)))  # Обработка всех строк по умолчанию
rows_to_process = [0] 
rows_to_process = 'all'
exp_n = 512
rows_to_process = np.asarray(df['experiment'] == exp_n).nonzero()[0].tolist()

if rows_to_process == 'all':
    rows_to_process = range(len(df))

# Функция для обработки каждого файла
def process_file(file_path, location):
    with czifile.CziFile(file_path) as czi:
        # Получение данных изображения
        image_data = czi.asarray()
        print("Image shape:", image_data.shape)
        
        # Пример доступа к данным изображения
        # В зависимости от формата файла может потребоваться изменить индексы
        channel_1 = 0
        channel_3 = 3
        
        if np.size(image_data, 2) == 3:
            channel_3 = 2
            print("3 channels instead of 4")
        
        slide = 2
        
        # Извлечение данных для каждого канала
        sample_slice_1 = image_data[0, 0, channel_1, slide, :, :, 0]
        sample_slice_3 = image_data[0, 0, channel_3, slide, :, :, 0]
        
        # Объединение изображений в одно с различными цветами
        # Создание пустого RGB изображения
        combined_image = np.zeros((*sample_slice_1.shape, 3), dtype=np.uint8)
        
        # Нормализация изображений для лучшего отображения
        sample_slice_1_normalized = (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1)) * 255
        sample_slice_3_normalized = (sample_slice_3 - np.min(sample_slice_3)) / (np.max(sample_slice_3) - np.min(sample_slice_3)) * 255
        
        # Назначение цветов
        combined_image[:, :, 0] = sample_slice_1_normalized.astype(np.uint8)  # Красный канал
        combined_image[:, :, 2] = sample_slice_3_normalized.astype(np.uint8)  # Синий канал
        
    # Определение разрешения из combined_image
    height, width = combined_image.shape[:2]
    dpi = height / 6.16
    
    # Визуализация объединенного изображения
    fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)  # Уменьшение размера окна
    ax.imshow(combined_image)
    # ax.set_title(f"{file_path} - {location}", fontsize=5)
    ax.axis('off')  # Убираем отображение осей

    # Определение координат для ROI, охватывающего всю область изображения
    coords = [(0, 0), (0, height), (width, height), (width, 0)]
    
    # Сохранение координат в файл
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    experiment_date = os.path.basename(os.path.dirname(file_path))
    coords_file_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_roi_coords.csv")
    
    coords_df = pd.DataFrame(coords, columns=['x', 'y'])
    coords_df.to_csv(coords_file_path, sep=';', index=False)
    print(f"Coordinates saved to {coords_file_path}")

    # Сохранение изображения с нарисованным ROI
    polygon = Polygon(coords, closed=True, edgecolor='#1DE720', facecolor='none', linewidth=2, alpha=0.7)
    ax.add_patch(polygon)
    
    # Добавление текста на изображение
    ax.text(0.5, 0.95, f"{file_path} - {location}", transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))
    
    # Сохранение изображения до отображения
    image_file_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_with_roi.png")
    plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)
    print(f"Image with ROI saved to {image_file_path}")

    # Показ изображения на одну секунду
    plt.show(block=False)
    # time.sleep(1)  # Пауза на одну секунду перед закрытием
    plt.close(fig)

# Обработка указанных строк
for idx in rows_to_process:
    file_path = df.iloc[idx]['filepath']
    location = df.iloc[idx]['location']
    process_file(file_path, location)

exp_n = 512
rows_to_process = np.asarray(df['experiment'] == exp_n).nonzero()[0].tolist()
#%% 2 Бинаризация с применением указанных ROI

import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from skimage import filters, measure, morphology
from matplotlib.path import Path
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"
rows_to_process = [0]  # Укажите номера строк для обработки (начиная с 0 для первой строки)


exp_n = 512
rows_to_process = np.asarray(df['experiment'] == exp_n).nonzero()[0].tolist()

rows_to_process = 'all'

if rows_to_process == 'all':
    rows_to_process = range(len(df))

pixel_to_micron_ratio = 0.1  # Коэффициент перевода из пикселей в микрометры (предположение)

# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')

error_files = []  # Список файлов, в которых возникли ошибки

# Инициализация прогресс-бара
with tqdm(total=len(rows_to_process), desc="Processing images") as pbar:
    for row_number_to_process in rows_to_process:
        try:
            # Получаем строку для обработки
            row = df.iloc[row_number_to_process]
            
            # Проверяем значение take_to_stat
            if row['take_to_stat'] == 'no':
                print(f"Skipping row {row_number_to_process} because take_to_stat is 'no'")
                pbar.update(1)  # Обновляем прогресс-бар
            else:
                image_path = row['filepath']
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                experiment_date = os.path.basename(os.path.dirname(image_path))
            
                # Пути к изображениям
                denoised_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
                masks_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.tif")
                full_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_full_roi_result_table.xls")
                summary_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xls")
                roi_coords_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_roi_coords.csv")
            
                # Чтение денойзированного изображения
                image = Image.open(denoised_image_path).convert('L')
                image_array = np.array(image)
            
                # Чтение координат ROI
                roi_coords_df = pd.read_csv(roi_coords_path, delimiter=';')
                roi_coords = roi_coords_df[['x', 'y']].values
                roi_path = Path(roi_coords)
            
                # Создание маски для ROI
                x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                roi_mask = roi_path.contains_points(points).reshape(image_array.shape)
            
                # Бинаризация изображения
                threshold_value = filters.threshold_otsu(image_array)
                binary_image = image_array > threshold_value
                binary_image = morphology.remove_small_objects(binary_image, min_size=64)
            
                # Применение маски ROI к бинаризованному изображению
                binary_image_roi = binary_image & roi_mask
            
                # Сохранение бинаризованного изображения
                binary_image_pil = Image.fromarray((binary_image_roi * 255).astype(np.uint8))
                binary_image_pil = ImageOps.invert(binary_image_pil)  # Инвертируем изображение
                binary_image_pil.save(masks_image_path)
            
                # Поиск объектов и измерение параметров
                labeled_image = measure.label(binary_image_roi)
                props = measure.regionprops(labeled_image, intensity_image=image_array)
            
                # Подготовка данных для сохранения
                results = []
                total_objects = 0
                total_area = 0
                total_mean_intensity = 0
                roi_area = np.sum(roi_mask) * pixel_to_micron_ratio**2  # Площадь ROI в микрометрах
                
                for index, prop in enumerate(props, start=1):
                    area_microns = prop.area * pixel_to_micron_ratio**2
                    total_objects += 1
                    total_area += area_microns
                    total_mean_intensity += prop.mean_intensity * area_microns
                    results.append({
                        "": index,  # Индексы объектов
                        "Area": f"{area_microns:.3f}",
                        "Mean": f"{prop.mean_intensity:.3f}",
                        "Min": int(prop.min_intensity),
                        "Max": int(prop.max_intensity)
                    })
            
                # Сохранение данных до суммирования
                results_df = pd.DataFrame(results)
                results_df.to_csv(full_result_path, sep='\t', index=False)
            
                # Подготовка суммарных данных
                if total_objects > 0:
                    average_size = total_area / total_objects
                    average_mean_intensity = total_mean_intensity / total_area
                else:
                    average_size = 0
                    average_mean_intensity = 0
            
                summary_result = {
                    "": 1,  # Пустая колонка с единицей в первой строке
                    "Slice": os.path.basename(masks_image_path),
                    "Count": total_objects,
                    "Total Area": f"{total_area:.3f}",
                    "Average Size": f"{average_size:.3f}",
                    "%Area": f"{(total_area / roi_area) * 100:.3f}",
                    "Mean": f"{average_mean_intensity:.3f}"
                }
            
                # Сохранение суммарных данных
                summary_df = pd.DataFrame([summary_result])
                summary_df.to_csv(summary_result_path, sep='\t', index=False)
            
                print(f"Row {row_number_to_process} processed. Saved binary image as {masks_image_path}.")
                pbar.update(1)  # Обновляем прогресс-бар
        
        except Exception as e:
            print(f"Error processing row {row_number_to_process} for file {image_path}: {e}")
            error_files.append(image_path)
            pbar.update(1)  # Обновляем прогресс-бар

# Вывод файлов, в которых возникли ошибки
if error_files:
    print("\nErrors occurred in the following files:")
    for error_file in error_files:
        print(error_file)
else:
    print("\nNo errors occurred during processing.")


#%% Бинаризация с применением указанных ROI max entropy threshold

import os
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
from skimage import filters, measure, morphology
from skimage.exposure import histogram
from matplotlib.path import Path
from tqdm import tqdm  # Импортируем tqdm для прогресс-бара

# Функция для вычисления порога по максимальной энтропии
def max_entropy_threshold(image):
    hist, bin_edges = histogram(image, nbins=256)
    hist = hist.astype(float)
    hist /= hist.sum()
    
    c_hist = hist.cumsum()
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    best_thr = bin_mids[0]
    max_ent = 0
    
    for i in range(len(hist)):
        p1 = hist[:i].sum()
        p2 = hist[i:].sum()
        
        if p1 == 0 or p2 == 0:
            continue
        
        h1 = -np.sum(hist[:i] / p1 * np.log2(hist[:i] / p1 + 1e-12))
        h2 = -np.sum(hist[i:] / p2 * np.log2(hist[i:] / p2 + 1e-12))
        
        ent = h1 + h2
        
        if ent > max_ent:
            max_ent = ent
            best_thr = bin_mids[i]
    
    return best_thr

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"
rows_to_process = [0]  # Укажите номера строк для обработки (начиная с 0 для первой строки)


exp_n = 512
rows_to_process = np.asarray(df['experiment'] == exp_n).nonzero()[0].tolist()

rows_to_process = 'all'

if rows_to_process == 'all':
    rows_to_process = range(len(df))

pixel_to_micron_ratio = 0.1  # Коэффициент перевода из пикселей в микрометры (предположение)

# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')

error_files = []  # Список файлов, в которых возникли ошибки

# Инициализация прогресс-бара
with tqdm(total=len(rows_to_process), desc="Processing images") as pbar:
    for row_number_to_process in rows_to_process:
        try:
            # Получаем строку для обработки
            row = df.iloc[row_number_to_process]
            
            # Проверяем значение take_to_stat
            if row['take_to_stat'] == 'no':
                print(f"Skipping row {row_number_to_process} because take_to_stat is 'no'")
                pbar.update(1)  # Обновляем прогресс-бар
            else:
                image_path = row['filepath']
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                experiment_date = os.path.basename(os.path.dirname(image_path))
            
                # Пути к изображениям
                denoised_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
                masks_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.tif")
                full_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_full_roi_result_table.xls")
                summary_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xls")
                roi_coords_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_roi_coords.csv")
            
                # Чтение денойзированного изображения
                image = Image.open(denoised_image_path).convert('L')
                image_array = np.array(image)
            
                # Чтение координат ROI
                roi_coords_df = pd.read_csv(roi_coords_path, delimiter=';')
                roi_coords = roi_coords_df[['x', 'y']].values
                roi_path = Path(roi_coords)
            
                # Создание маски для ROI
                x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                roi_mask = roi_path.contains_points(points).reshape(image_array.shape)
            
                # Бинаризация изображения с использованием максимальной энтропии
                threshold_value = max_entropy_threshold(image_array)
                binary_image = image_array > threshold_value
                binary_image = morphology.remove_small_objects(binary_image, min_size=64)
            
                # Применение маски ROI к бинаризованному изображению
                binary_image_roi = binary_image & roi_mask
            
                # Сохранение бинаризованного изображения
                binary_image_pil = Image.fromarray((binary_image_roi * 255).astype(np.uint8))
                binary_image_pil = ImageOps.invert(binary_image_pil)  # Инвертируем изображение
                binary_image_pil.save(masks_image_path)
            
                # Поиск объектов и измерение параметров
                labeled_image = measure.label(binary_image_roi)
                props = measure.regionprops(labeled_image, intensity_image=image_array)
            
                # Подготовка данных для сохранения
                results = []
                total_objects = 0
                total_area = 0
                total_mean_intensity = 0
                roi_area = np.sum(roi_mask) * pixel_to_micron_ratio**2  # Площадь ROI в микрометрах
                
                for index, prop in enumerate(props, start=1):
                    area_microns = prop.area * pixel_to_micron_ratio**2
                    total_objects += 1
                    total_area += area_microns
                    total_mean_intensity += prop.mean_intensity * area_microns
                    results.append({
                        "": index,  # Индексы объектов
                        "Area": f"{area_microns:.3f}",
                        "Mean": f"{prop.mean_intensity:.3f}",
                        "Min": int(prop.min_intensity),
                        "Max": int(prop.max_intensity)
                    })
            
                # Сохранение данных до суммирования
                results_df = pd.DataFrame(results)
                results_df.to_csv(full_result_path, sep='\t', index=False)
            
                # Подготовка суммарных данных
                if total_objects > 0:
                    average_size = total_area / total_objects
                    average_mean_intensity = total_mean_intensity / total_area
                else:
                    average_size = 0
                    average_mean_intensity = 0
            
                summary_result = {
                    "": 1,  # Пустая колонка с единицей в первой строке
                    "Slice": os.path.basename(masks_image_path),
                    "Count": total_objects,
                    "Total Area": f"{total_area:.3f}",
                    "Average Size": f"{average_size:.3f}",
                    "%Area": f"{(total_area / roi_area) * 100:.3f}",
                    "Mean": f"{average_mean_intensity:.3f}"
                }
            
                # Сохранение суммарных данных
                summary_df = pd.DataFrame([summary_result])
                summary_df.to_csv(summary_result_path, sep='\t', index=False)
            
                print(f"Row {row_number_to_process} processed. Saved binary image as {masks_image_path}.")
                pbar.update(1)  # Обновляем прогресс-бар
        
        except Exception as e:
            print(f"Error processing row {row_number_to_process} for file {image_path}: {e}")
            error_files.append(image_path)
            pbar.update(1)  # Обновляем прогресс-бар

# Вывод файлов, в которых возникли ошибки
if error_files:
    print("\nErrors occurred in the following files:")
    for error_file in error_files:
        print(error_file)
else:
    print("\nNo errors occurred during processing.")
#%% постпроцессинг и модификация таблицы с данными c ROI

import pandas as pd
import os
import re
from PIL import Image

# Путь к CSV файлу с путями к изображениям
csv_file_path = "E:\\iMAGES\\protocol.csv"
output_directory = "E:\\iMAGES\\processed_images_full"

# Создаем выходную директорию, если ее нет
os.makedirs(output_directory, exist_ok=True)

# Загружаем данные из CSV
df = pd.read_csv(csv_file_path, delimiter=';')
df = df[df['take_to_stat'] != 'no']

# Функция для объединения изображений и сохранения их
def combine_and_save_images(image1_path, image2_path, output_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    
    # Определяем размеры нового изображения
    combined_width = image1.width + image2.width
    combined_height = max(image1.height, image2.height)
    
    # Создаем новое изображение
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (image1.width, 0))
    
    # Сохраняем объединенное изображение
    combined_image.save(output_path)

# Коллекционируем данные и объединяем изображения
summary_data_list = []

for index, row in df.iterrows():
    image_path = row['filepath']
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    experiment_date = os.path.basename(os.path.dirname(image_path))
    
    # Пути к изображениям
    denoised_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
    masks_image_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.tif")

    # Путь к файлу с результатами
    summary_result_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xls")
    roi_coords_path = os.path.join(os.path.dirname(image_path), f"{experiment_date}_{base_name}_roi_coords.csv")

    # Объединяем и сохраняем изображения
    combined_image_path = os.path.join(output_directory, f"{experiment_date}_{base_name}_combined.tif")
    try:
        combine_and_save_images(denoised_image_path, masks_image_path, combined_image_path)
    except Exception as e:
        print(f"Error combining images for {image_path}: {e}")
        continue

    # Сохраняем первую строку таблицы результатов
    summary_data = pd.read_csv(summary_result_path, sep='\t')
    if summary_data is not None:
        summary_data['filepath'] = image_path
        summary_data['location'] = row['location']
        summary_data['threshold_method'] = row['threshold_method']
        summary_data_list.append(summary_data)

# Создаем DataFrame с коллекционированными данными
summary_df = pd.concat(summary_data_list, ignore_index=True)
summary_df.drop(summary_df.columns[[0, -1]], axis=1, inplace=True)

summary_df['Postnatal_Age'] = np.asarray(df['Postnatal_Age'])

summary_df['Experiment_Number'] = np.asarray(df['Experiment_Number'])

# Сохранение обновленного DataFrame в новый файл
summary_output_path = os.path.join(output_directory, "collected_roi_summary_data.xlsx")
summary_df.to_excel(summary_output_path, index=False)

print("Постпроцессинг завершен. Обновленные данные сохранены в:", summary_output_path)

#
# # Функция для парсинга данных из строки
# def parse_slice_info(slice_info):
#     pattern = r"P(\d+)(?:\.\d+)*\s+slide(\d+)\s+slice(\d+)_Experiment-(\d+)"
#     match = re.search(pattern, slice_info)
#     if match:
#         postnatal_age = int(match.group(1))  # берем только целое число
#         slide_number = int(match.group(2))
#         slice_number = int(match.group(3))
#         experiment_number = int(match.group(4))
#         return postnatal_age, slide_number, slice_number, experiment_number
#     return None, None, None, None

# # Добавление новых колонок в DataFrame
# summary_df['Postnatal_Age'] = None
# summary_df['Slide_Number'] = None
# summary_df['Slice_Number'] = None
# summary_df['Experiment_Number'] = None

# # Парсинг данных и заполнение новых колонок
# for index, row in summary_df.iterrows():
#     slice_info = row['Slice']
#     postnatal_age, slide_number, slice_number, experiment_number = parse_slice_info(slice_info)
#     summary_df.at[index, 'Postnatal_Age'] = postnatal_age
#     summary_df.at[index, 'Slide_Number'] = slide_number
#     summary_df.at[index, 'Slice_Number'] = slice_number
#     summary_df.at[index, 'Experiment_Number'] = experiment_number


#%% статистика с виолинами c ROI


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
import numpy as np

# Load the data
file_path = 'E:/iMAGES/processed_images_full/collected_roi_summary_data.xlsx'
data = pd.read_excel(file_path)

# Plotting function
def plot_numerical_parameters(data, parameters, category, hue):
    for parameter in parameters:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=category, y=parameter, hue=hue, data=data, palette="deep")
        plt.title(f'Relationship between {parameter} and {category} by {hue}')
        plt.xlabel(category)
        plt.ylabel(parameter)
        plt.legend(title=hue)
        plt.show()

# List of numerical parameters
# numerical_parameters = ['Count', 'Total Area', 'Average Size', '%Area', 'Mean'] % оставь в комментарии
numerical_parameters = ['%Area']

# Plot the data with respect to Postnatal_Age and color-coded by location
# plot_numerical_parameters(data, numerical_parameters, 'Postnatal_Age', 'location')

# Filtering the data for Postnatal_Age 5, 11, and 15
filtered_data = data[data['Postnatal_Age'].isin([5, 11, 15, 21])]

# Violin plot function
def plot_violin(data, parameter, category, hue):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=category, y=parameter, hue=hue, data=data, fill=False)
    plt.title(f'Violin plot of {parameter} by {category} and {hue}')
    plt.xlabel(category)
    plt.ylabel(parameter)
    plt.legend(title=hue)
    plt.show()

# Plotting violin plots for the numerical parameters
for parameter in numerical_parameters:
    plot_violin(filtered_data, parameter, 'Postnatal_Age', 'location')

# Performing statistical comparisons
stat_results = []
for parameter in numerical_parameters:
    age_groups = [filtered_data[filtered_data['Postnatal_Age'] == age][parameter] for age in [5, 11, 15]]
    stat, p = kruskal(*age_groups)
    stat_results.append({'Parameter': parameter, 'Kruskal-Wallis H': stat, 'p-value': p})

# Creating a dataframe for the statistical results
stat_results_df = pd.DataFrame(stat_results)

# Function to calculate pairwise p-values using Mann-Whitney U test
def pairwise_comparisons(data, parameter, groups, group_col='Postnatal_Age'):
    p_values = np.zeros((len(groups), len(groups)))
    
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:
                group1_data = data[data[group_col] == group1][parameter]
                group2_data = data[data[group_col] == group2][parameter]
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                p_values[i, j] = p_value
                p_values[j, i] = p_value
            elif i == j:
                p_values[i, j] = np.nan
    
    p_values_df = pd.DataFrame(p_values, index=groups, columns=groups)
    return p_values_df

# List of groups to compare
groups = [5, 11, 15, 21]

# Initialize a dictionary to store pairwise p-values for each parameter
pairwise_p_values = {}

# Calculate pairwise comparisons for each parameter
for parameter in numerical_parameters:
    pairwise_p_values[parameter] = pairwise_comparisons(filtered_data, parameter, groups)

# Display the pairwise p-values
for parameter, p_values_df in pairwise_p_values.items():
    print(f"Pairwise p-values for {parameter}:")
    print(p_values_df)
#%% создаем исполняемый файл
import subprocess
import sys
import os

# Путь к вашему интерпретатору Python
python_interpreter = 'C:\\Users\\ta3ma\\anaconda3\\python.exe'

# Команда для создания exe файла с помощью PyInstaller
command = [
    python_interpreter,
    '-m', 'PyInstaller',
    '--onefile',
    '--distpath', 'C:\\Users\\ta3ma\\Documents\\synapto_catch',
    'C:\\Users\\ta3ma\\Documents\\synapto_catch\\select_roi.py'
]

# Запуск команды
subprocess.run(command, check=True)

print("EXE файл успешно создан!")
