##%matplotlib qt
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
summary_df['ID'] = None

# Парсинг данных и заполнение новых колонок
for index, row in summary_df.iterrows():
    slice_info = row['Slice']
    postnatal_age, slide_number, slice_number, experiment_number = parse_slice_info(slice_info)
    summary_df.at[index, 'Postnatal_Age'] = postnatal_age
    summary_df.at[index, 'Slide_Number'] = slide_number
    summary_df.at[index, 'Slice_Number'] = slice_number
    summary_df.at[index, 'ID'] = experiment_number

# Сохранение обновленного DataFrame в новый файл
summary_output_path = os.path.join(output_directory, "collected_summary_data.xlsx")
summary_df.to_excel(summary_output_path, index=False)

print("Постпроцессинг завершен. Обновленные данные сохранены в:", summary_output_path)

#%% денойзинг
##%matplotlib qt
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

##%matplotlib qt

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
rows_to_process = np.asarray(df['ID'].isin(exp_list)).nonzero()[0].tolist()

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

summary_df['ID'] = np.asarray(df['ID'])

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
#         return postnatal_age, slide_number, slice_number, ID
#     return None, None, None, None

# # Добавление новых колонок в DataFrame
# summary_df['Postnatal_Age'] = None
# summary_df['Slide_Number'] = None
# summary_df['Slice_Number'] = None
# summary_df['ID'] = None

# # Парсинг данных и заполнение новых колонок
# for index, row in summary_df.iterrows():
#     slice_info = row['Slice']
#     postnatal_age, slide_number, slice_number, experiment_number = parse_slice_info(slice_info)
#     summary_df.at[index, 'Postnatal_Age'] = postnatal_age
#     summary_df.at[index, 'Slide_Number'] = slide_number
#     summary_df.at[index, 'Slice_Number'] = slice_number
#     summary_df.at[index, 'ID'] = experiment_number


# %% статистика с виолинами c ROI - все на одном графике
##%matplotlib qt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import numpy as np
from itertools import combinations

# Load the data
file_path = 'E:/iMAGES/results/collected_roi_summary_data.xlsx'
data = pd.read_excel(file_path)
data = data[data['location'] != 'SP_CA3']
# List of numerical parameters
numerical_parameters = ['%Area']
output_folder = 'C:/Users/ta3ma/Dropbox/RN/PhDs/PhD Azat/PhD_text/material/Synaptotagmin/media/'

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

def plot_pairwise_pvalues(local_param_data, groups, local_p_values, groups_x_pos):
    height = local_param_data.quantile(0.95)
    h_step = (local_param_data.quantile(0.95) * 2 - height) / (len(groups) + 1)
    
    for j, group1 in enumerate(groups):
        for k, group2 in enumerate(groups):
            if j < k:
                p_value = local_p_values.loc[group1, group2]
                ranktext = rankstars(p_value)
                if not np.isnan(p_value) and ranktext != 'ns':
                    x1, x2 = groups_x_pos[j], groups_x_pos[k]
                    y, h, col = local_param_data.max() + 1, 1, (0.5, 0.5, 0.5, 0.5)
                    height = height + h_step
                    plt.plot([x1, x2], [height, height], lw=1.5, c=col)
                    plt.plot([x2, x2], [height - h_step * 0.3, height], lw=1.5, c=col)
                    plt.plot([x1, x1], [height - h_step * 0.3, height], lw=1.5, c=col)
                    plt.text((x1 + x2) * 0.5, height, ranktext, ha='center', va='bottom', color='k')


# Helper function for ranking stars
def rankstars(pval):
    if pval < 0.0001:
        return '****'
    elif pval < 0.001:
        return '***'
    elif pval < 0.01:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return 'ns'

# Filtering the data for Postnatal_Age 5, 11, 15, and 21
filtered_data = data[data['Postnatal_Age'].isin([5, 11, 15, 21])]

# Violin plot function
def plot_violin(data, parameter, category, hue):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x=category, y=parameter, hue=hue, data=data, fill=False)
    # plt.title(f'Violin plot of {parameter} by {category} and {hue}')
    plt.xlabel(category.replace('_', ' '))
    plt.ylabel(parameter)
    plt.legend(title=hue)
    plt.show()
    # Remove the box around the plot
    sns.despine()

def calculate_positions(start_index, num_groups, width = 0.8):
    # Determine the centre of the index
    centre = start_index + width / (2*num_groups)    
    # Determine the offset to centre the groups
    offset = width / (num_groups)    
    # Calculate the positions of the groups
    positions = [centre - (num_groups / 2 - i) * offset for i in range(num_groups)]    
    return positions

# Perform statistical tests
age_groups = [5, 11, 15, 21]
results = []
p_values_dict = {}

for age in age_groups:
    age_data = filtered_data[filtered_data['Postnatal_Age'] == age]
    locations = age_data['location'].unique()
    p_values_matrix = pd.DataFrame(index=locations, columns=locations, dtype=float)
    
    for (loc1, loc2) in combinations(locations, 2):
        group1 = age_data[age_data['location'] == loc1][numerical_parameters[0]]
        group2 = age_data[age_data['location'] == loc2][numerical_parameters[0]]
        stat, p_value = mannwhitneyu(group1, group2)
        results.append({'Postnatal_Age': age, 'Location_Comparison': f'{loc1} vs {loc2}', 'Test': 'Mann-Whitney U', 'p_value': p_value})
        p_values_matrix.loc[loc1, loc2] = p_value
        p_values_matrix.loc[loc2, loc1] = p_value
    
    p_values_dict[age] = p_values_matrix

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)


# Plotting violin plots for the numerical parameters
for parameter in numerical_parameters:
    plot_violin(filtered_data, parameter, 'Postnatal_Age', 'location')
    for age_index, age in enumerate(age_groups):
        age_data = filtered_data[filtered_data['Postnatal_Age'] == age]
        locations = age_data['location'].unique()
        local_param_data = age_data[parameter]
        local_p_values = p_values_dict[age]
        groups_x_pos = calculate_positions(age_index, np.size(locations))
        plot_pairwise_pvalues(local_param_data, locations, local_p_values, groups_x_pos)


# plt.savefig(f'{output_folder}inter_age_violin_plot.png')
# %%

#%% статистика с виолинами c ROI - Для каждой локации свой график

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
import scikit_posthocs as sp
import numpy as np
import os

output_folder = 'C:/Users/ta3ma/Dropbox/RN/PhDs/PhD Azat/PhD_text/material/Synaptotagmin/media/'
# Load the data
file_path = 'E:/iMAGES/results/collected_roi_summary_data.xlsx'
data = pd.read_excel(file_path)

# List of numerical parameters
numerical_parameters = ['%Area']

# Filtering the data for Postnatal_Age 5, 11, 15, and 21
filtered_data = data[data['Postnatal_Age'].isin([5, 11, 15, 21, 120])]

# Function to calculate Kruskal-Wallis p-values for each location
def calculate_kruskal_pvalues(data, parameter, groups, group_col='Postnatal_Age', location_col='location'):
    results = {}
    for location in data[location_col].unique():
        group_data = [data[(data[group_col] == group) & (data[location_col] == location)][parameter].dropna() for group in groups]
        if all(len(g) > 0 for g in group_data):
            stat, p_value = kruskal(*group_data)
            results[location] = p_value
        else:
            results[location] = np.nan
    return results

# Function to perform pairwise comparisons using Dunn's test
def calculate_dunn_pvalues(data, parameter, groups, group_col='Postnatal_Age', location_col='location'):
    pairwise_results = {}
    for location in data[location_col].unique():
        location_data = data[data[location_col] == location]
        dunn_results = sp.posthoc_dunn(location_data, val_col=parameter, group_col=group_col, p_adjust='bonferroni')
        pairwise_results[location] = dunn_results
    return pairwise_results

# Function to convert p-values to stars
def rankstars(p):
    if not np.isnan(p):
        if p > 0.05:
            return 'ns'
        elif p <= 0.0001:
            return '****'
        elif p <= 0.001:
            return '***'
        elif p <= 0.01:
            return '**'
        elif p <= 0.05:
            return '*'
    else:
        return 'ns'

# Function to map location names
def map_location_name(location):
    if location == 'SO':
        return 'Stratum Oriens'
    elif location == 'SP_CA1':
        return 'Stratum Pyramidale CA1'
    elif location == 'SP_CA3':
        return 'Stratum Pyramidale CA3'
    elif location == 'SR':
        return 'Stratum Radiatum'
    return location

def plot_pairwise_pvalues(local_param_data, groups, local_p_values, groups_x_pos):
    height = local_param_data.quantile(0.95)
    h_step = (local_param_data.quantile(0.95) * 2 - height) / (len(groups) + 1)
    
    for j, group1 in enumerate(groups):
        for k, group2 in enumerate(groups):
            if j < k:
                p_value = local_p_values.loc[group1, group2]
                ranktext = rankstars(p_value)
                if not np.isnan(p_value) and ranktext != 'ns':
                    x1, x2 = groups_x_pos[j], groups_x_pos[k]
                    y, h, col = local_param_data.max() + 1, 1, (0.5, 0.5, 0.5, 0.5)
                    height = height + h_step
                    plt.plot([x1, x2], [height, height], lw=1.5, c=col)
                    plt.plot([x2, x2], [height - h_step * 0.3, height], lw=1.5, c=col)
                    plt.plot([x1, x1], [height - h_step * 0.3, height], lw=1.5, c=col)
                    plt.text((x1 + x2) * 0.5, height, ranktext, ha='center', va='bottom', color='k')

# Function to plot violins with different colors for each figure and add p-values
def plot_violin_with_pvalues(data, parameter, category, hue, dunn_pvalues):
    locations = data[hue].unique()
    palette = sns.color_palette("deep", len(locations))
    groups = data[category].unique()
    groups.sort()
    for i, location in enumerate(locations):
        
        local_data = data[data[hue] == location]
        group_indexes = {group: index for index, group in enumerate(groups)}
        
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=category, y=parameter, data=local_data, color=palette[i], fill=False)
        
        scatter_x = local_data[category].map(group_indexes)
        scatter_x = scatter_x + np.random.uniform(-0.2, 0.2, size=len(scatter_x))
        sns.scatterplot(x=scatter_x, y=parameter, data=local_data, color=palette[i])
        
        plt.title(map_location_name(location))
        plt.xlabel(category.replace('_', ' '))
        plt.ylabel(parameter)
        local_p_values = dunn_pvalues[location]
        local_param_data = local_data[parameter]
        groups_x_pos = list(range(np.size(groups)))
        plot_pairwise_pvalues(local_param_data, groups, local_p_values, groups_x_pos)

        # Remove the box around the plot
        sns.despine()

        # Save the figure
        plt.savefig(f'{output_folder}{location}_violin_plot.png')
        # plt.close()

# List of groups to compare
groups = [5, 11, 15, 21, 120]

# Calculate Kruskal-Wallis p-values for each parameter and each location
kruskal_p_values = {}
for parameter in numerical_parameters:
    kruskal_p_values[parameter] = calculate_kruskal_pvalues(filtered_data, parameter, groups)

# Calculate Dunn's test pairwise p-values for each parameter and each location
dunn_p_values = {}
for parameter in numerical_parameters:
    dunn_p_values[parameter] = calculate_dunn_pvalues(filtered_data, parameter, groups)

# Plotting violin plots for the numerical parameters with Kruskal-Wallis and Dunn's test p-values
for parameter in numerical_parameters:
    dunn_pvalues = dunn_p_values[parameter]
    plot_violin_with_pvalues(filtered_data, parameter, 'Postnatal_Age', 'location', dunn_pvalues)

# Display the Kruskal-Wallis and Dunn's test p-values for each location
for parameter, location_results in kruskal_p_values.items():
    print(f"Kruskal-Wallis p-values for {parameter}:")
    for location, p_value in location_results.items():
        print(f"Location: {location}, p-value: {p_value}")

for parameter, location_results in dunn_p_values.items():
    print(f"Dunn's test pairwise p-values for {parameter}:")
    for location, p_values_df in location_results.items():
        print(f"Location: {location}")
        print(p_values_df)
#%% определяем ось
##%matplotlib qt

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageOps import invert
from matplotlib.widgets import Cursor

# Функция для поворота изображения вокруг заданной точки без изменения размера
def rotate_image(image, angle, center):
    image = image.convert('RGBA')  # Преобразуем в формат с альфа-каналом
    rotated_image = Image.new('RGBA', image.size)
    rotated_image.paste(image.rotate(angle, resample=Image.BICUBIC, center=center), (0, 0), image.rotate(angle, resample=Image.BICUBIC, center=center))
    return rotated_image.convert('RGB')

# Функция для вычисления угла линии
def calculate_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

# Функция для поворота координат относительно заданного центра
def rotate_coords(x, y, angle, cx, cy):
    angle_rad = np.radians(angle)
    x_new = cx + (x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad)
    y_new = cy + (x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad)
    return x_new, y_new

# Функция для отображения изображения с точками
def show_image(image, points=None, title=""):
    height = image.height
    width = image.width
    dpi = 200
    figsize = width / float(dpi), height / float(dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы
        
    ax.imshow(image, cmap='gray')
    ax.axis('off')

    if points is not None:
        x_coords, y_coords = zip(*points)
        ax.plot(x_coords, y_coords, 'ro')

    ax.set_title(title)
    plt.show()
    return fig, ax

def calculate_brightness_histogram(image, bin_size):

    # Преобразовать изображение в оттенки серого
    gray_image = image.convert('L')
    
    # Преобразовать изображение в массив numpy
    gray_array = np.array(gray_image)
    
    # Определить количество строк и столбцов в изображении
    height, width = gray_array.shape
    
    # Инициализировать список для хранения средней яркости блоков
    average_brightness = []
    
    # Вычислить среднюю яркость для каждого блока
    for i in range(0, height, bin_size):
        block = gray_array[i:i+bin_size, :]
        block_mean = np.mean(block)
        average_brightness.append(block_mean)
        
    return average_brightness
    
# Открытие изображения
image_path = "E:\\iMAGES\\SP P21 3\\SP P21 3_Experiment-428_masks_roi_crop.png"
image = Image.open(image_path)
image = invert(image)

# Отображение изображения и получение координат линии от пользователя
fig, ax = show_image(image, title="Select two points")

cursor = Cursor(ax, useblit=True, color='red', linewidth=2)

coords = []
cropped_image = None

def onclick(event):
    global cropped_image
    if event.xdata is not None and event.ydata is not None:
        coords.append((event.xdata, event.ydata))
        if len(coords) == 4:
            fig.canvas.mpl_disconnect(cid)
            plt.close(fig)
            
            (x1, y1), (x2, y2), (x3, y3), (x4, y4) = coords
            angle = calculate_angle(x1, y1, x2, y2)
            
            # Поворот изображения вокруг второй точки
            rotated_image = rotate_image(image, angle, center=(x2, y2))
            
            # Поворот координат точек
            new_coords = []
            for (x, y) in [(x1, y1), (x3, y3), (x4, y4)]:
                new_coords.append(rotate_coords(x, y, -angle, x2, y2))
            
            (x1_new, y1_new), (x3_new, y3_new), (x4_new, y4_new) = new_coords

            
            # Отображение повернутого изображения с повернутой первой точкой
            # show_image(rotated_image, points=[(x1_new, y1_new), (x2, y2)], title="Rotated Image with Points")
            
            # Обрезка изображения по x1_new, x2, y1_new, и y2
            left = min(x1_new, x2)
            right = max(x1_new, x2)
            
            top = min(y3_new, y4_new)  # min(y1_new, y2)
            bottom = y2  # rotated_image.height  # max(y1_new, y2)
            print(y1_new)
            cropped_image = rotated_image.crop((left, top, right, bottom))
            
            # Отображение повернутого и обрезанного изображения с повернутой первой точкой
            # show_image(cropped_image, points=[(x1_new - left, y1_new), (x2 - left, y2)], title="Rotated and Cropped Image with Points")
            
            return cropped_image
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

#%% Если изображение было обрезано, вычислить и построить гистограмму яркости

show_image(cropped_image)
bin_size = 1
average_brightness = calculate_brightness_histogram(cropped_image.rotate(180), bin_size)

# Построить гистограмму
plt.figure(figsize=(10, 6))
plt.plot(average_brightness)
plt.title('Гистограмма яркости вдоль горизонтальной оси')
plt.xlabel('Блоки строк изображения')
plt.ylabel('Средняя яркость')
plt.show()

#%% semi-auto radial axis

#%matplotlib qt
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
from os.path import splitext, basename, dirname, join, exists
from matplotlib.widgets import Cursor

def show_image(image, title="", points=None, point_names=None, lines=None):
    height = image.height
    width = image.width
    dpi = 200
    figsize = width / float(dpi), height / float(dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы

    ax.imshow(image, cmap='gray')
    ax.axis('off')

    if points:
        x_coords, y_coords = zip(*points)
        ax.plot(x_coords, y_coords, 'ro')
        if point_names:
            for (x, y), name in zip(points, point_names):
                ax.text(x, y, name, fontsize=5, color='white', ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

    if lines:
        for (x1, y1), (x2, y2), name in lines:
            ax.plot([x1, x2], [y1, y2], 'r-')
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, name, fontsize=5, color='white', ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5))

    ax.set_title(title)
    # Добавление текста на изображение
    ax.text(0.5, 0.98, title, transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))
    plt.show()
    return fig, ax

# Функция для обработки клика мыши

def create_graphic_functions(graph_element_type, comment, filepaths, points, point_names, lines):
    
    (csv_path, base_name, experiment_date) = filepaths
    # points, point_names, lines = load_comments(csv_path)
    
    coords = []
    def addLine(event):
        if event.xdata is not None and event.ydata is not None:
            coords.append((event.xdata, event.ydata))
            if len(coords) == 2:
                (x1, y1), (x2, y2) = coords
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                
                save_comment(x1, y1, comment, 'line', csv_path)
                save_comment(x2, y2, comment, 'line', csv_path)
    
                lines.append(((x1, y1), (x2, y2), comment))
    
                plt.close()
                fig, ax = show_image(img, points=points, point_names=point_names, lines=lines)
                image_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_comments.png")
                plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f"Image saved to {image_file_path}")
                coords.clear()
    def addPoint(event):
        if event.xdata and event.ydata:
            x, y = int(event.xdata), int(event.ydata)
            save_comment(x, y, comment, 'point', csv_path)
            points.append((x, y))
            point_names.append(comment)
            plt.close()
            fig, ax = show_image(img, points=points, point_names=point_names, lines=lines)
            image_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_comments.png")
            plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Image with ROI saved to {image_file_path}")
            
    return addPoint if graph_element_type == 'point' else addLine

def save_comment(x, y, comment, point_type, csv_path):
    data = {"x": [x], "y": [y], "comment": [comment], "type": [point_type]}
    df = pd.DataFrame(data)
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(csv_path, index=False)

def load_comments(csv_path):
    points = []
    point_names = []
    lines = []

    # Загружаем существующие комментарии, если они есть
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        line_coords = []
        for _, row in df_existing.iterrows():
            x, y, comment, point_type = row["x"], row["y"], row["comment"], row["type"]
            if point_type == 'point':
                points.append((x, y))
                point_names.append(comment)
            elif point_type == 'line':
                line_coords.append((x, y))
                if len(line_coords) == 2:
                    lines.append((line_coords[0], line_coords[1], comment))
                    line_coords = []
    return points, point_names, lines



def prepare_data(protocol_path, exps_input):
    df = pd.read_excel(protocol_path)
    # Filter out rows where 'take_to_stat' is 'no' and log removed experiments
    removed_experiments = df[df['take_to_stat'] == 'no']['ID'].tolist()
    df = df[df['take_to_stat'] != 'no']
    df.reset_index(drop=True, inplace=True)
    
    exps_input = parce_exps(exps_input)
    
    # Find rows with the specified experiment numbers
    if exps_input == 'all':
        rows_to_process = list(range(len(df)))
    else:
        rows_to_process = df[df['ID'].isin(exps_input)].index.tolist()
        
    return df, rows_to_process

def parce_exps(exps_input):
    exps_input = exps_input.strip().lower()
    if exps_input == 'all':
        return 'all'
    # Check if input is in the range format
    if ':' in exps_input:
        start, end = map(int, exps_input.split(':'))
        return list(range(start, end + 1))
    else:
        return [int(exp.strip()) for exp in exps_input.split(',')]

def display_image_with_cursor(comment, filepaths, img, graph_element_type):
    csv_path = filepaths[0]
    points, point_names, lines = load_comments(csv_path)
    fig, ax = show_image(img, title='put ' + comment, points=points, point_names=point_names, lines=lines)
    # cursor = Cursor(ax, useblit=True, color='red', linewidth=2)
    
    addGraphicElement = create_graphic_functions(graph_element_type, comment, filepaths, points, point_names, lines)
    cid = fig.canvas.mpl_connect('button_press_event', addGraphicElement)
    plt.ion()
    plt.show(block=True)
    print('Next')
    
exps_input = '867:11300'
protocol_path = "E:\iMAGES\protocol.xlsx"
df, rows_to_process = prepare_data(protocol_path, exps_input)
for idx, row_idx in enumerate(rows_to_process):
    file_path = df.iloc[row_idx]['filepath']
    location = df.iloc[row_idx]['location']   
    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    csv_path = join(dirname(file_path), f"{experiment_date}_{base_name}_comments.csv")
    
    image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_with_roi.png")
    img = Image.open(image_path)
    
    filepaths = (csv_path, base_name, experiment_date)
    # Display images with different comments and functions
    display_image_with_cursor('pyramidale', filepaths, img, 'point')
    display_image_with_cursor('radiatum', filepaths, img, 'point')
    display_image_with_cursor('oriens', filepaths, img, 'point')
    
    # For radial axis with different function
    display_image_with_cursor('radial axis', filepaths, img, 'line')




#%% автоматически преобразуем вдоль радиальной оси

#%matplotlib qt

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from math import atan2, degrees, radians, sin, cos
from matplotlib.patches import Polygon
from os.path import basename, dirname, join, splitext
from scipy.spatial import ConvexHull
from PIL.ImageOps import invert
from itertools import combinations
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm
from skimage.filters import threshold_otsu

# Функция для загрузки точек и линий из таблицы
def load_comments(csv_path):
    points = []
    point_names = []
    lines = []
    # Загружаем существующие комментарии, если они есть
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        line_coords = []
        for _, row in df_existing.iterrows():
            x, y, comment, point_type = row["x"], row["y"], row["comment"], row["type"]
            if point_type == 'point':
                points.append((x, y))
                point_names.append(comment)
            elif point_type == 'line':
                line_coords.append((x, y))
                if len(line_coords) == 2:
                    lines.append((line_coords[0], line_coords[1], comment))
                    line_coords = []
    return points, point_names, lines

# Функция для отображения изображения с точками и линиями
def show_image(image, title="", points=None, point_names=None, lines=None):
    height = image.height
    width = image.width
    dpi = 200
    figsize = width / float(dpi), height / float(dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    if points:
        x_coords, y_coords = zip(*points)
        ax.plot(x_coords, y_coords, 'ro')
        if point_names:
            for (x, y), name in zip(points, point_names):
                ax.text(x, y, name, fontsize=5, color='white', ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5))
    if lines:
        for (x1, y1), (x2, y2), name in lines:
            ax.plot([x1, x2], [y1, y2], 'y-')
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, name, fontsize=5, color='white', ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.5))
    # Добавление текста на изображение
    ax.text(0.5, 0.98, title, transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))
    plt.show()
    return fig, ax

# Функция для вычисления угла линии
def calculate_angle(line):
    (x1, y1), (x2, y2), _ = line
    angle = atan2(y2 - y1, x2 - x1)
    return degrees(angle)

# Функция для поворота точек
def rotate_point(point, angle, center):
    angle_rad = radians(angle)
    x, y = point
    cx, cy = center
    new_x = cx + (x - cx) * cos(angle_rad) - (y - cy) * sin(angle_rad)
    new_y = cy + (x - cx) * sin(angle_rad) + (y - cy) * cos(angle_rad)
    return new_x, new_y

# Функция для поворота всех точек и линий
def rotate_annotations(points, lines, angle, center):
    rotated_points = [rotate_point(point, angle, center) for point in points]
    rotated_lines = [(rotate_point(line[0], angle, center), rotate_point(line[1], angle, center), line[2]) for line in lines]
    return rotated_points, rotated_lines

# Функция для расширения изображения
def expand_image(image, angle, rotation_center):
    width, height = image.size
    cx, cy = rotation_center
    # Вычисление расстояний от центра вращения до краев изображения
    distances = [
        np.hypot(cx, cy),
        np.hypot(cx, height - cy),
        np.hypot(width - cx, cy),
        np.hypot(width - cx, height - cy)
    ]
    # Выбор максимального расстояния для расчета новой размерности изображения
    max_distance = max(distances)
    new_width = int(4 * max_distance)
    new_height = int(4 * max_distance)
    expanded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    shift_x = (new_width - width) // 2
    shift_y = (new_height - height) // 2
    expanded_image.paste(image, (shift_x, shift_y))
    new_center = (cx + shift_x, cy + shift_y)
    return expanded_image, new_center, shift_x, shift_y

# Функция для смещения точек и линий
def shift_annotations(points, lines, shift_x, shift_y):
    shifted_points = [(x + shift_x, y + shift_y) for x, y in points]
    shifted_lines = [((x1 + shift_x, y1 + shift_y), (x2 + shift_x, y2 + shift_y), name) for (x1, y1), (x2, y2), name in lines]
    return shifted_points, shifted_lines

# Функция для поворота изображения
def rotate_image(image, angle, image_center):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(np.array(image), rot_mat, image.size, flags=cv2.INTER_LINEAR)
    return Image.fromarray(rotated_image)

# Функция для поворота полигона
def rotate_polygon(polygon_coords, angle, center):
    return [rotate_point(point, angle, center) for point in polygon_coords]

def crop_rotated_image_with_annotations(rotated_image, rotated_points, rotated_lines, rotated_roi_coords, padding_ratio=0.1):
    # Найти границы всех точек и линий
    all_x_coords = [x for x, y in rotated_points]
    all_y_coords = [y for x, y in rotated_points]    
    for (x1, y1), (x2, y2), _ in rotated_lines:
        all_x_coords.extend([x1, x2])
        all_y_coords.extend([y1, y2])
    # Добавить координаты полигона
    all_x_coords += [x for x, y in rotated_roi_coords]
    all_y_coords += [y for x, y in rotated_roi_coords]
    # Найти минимальные и максимальные координаты
    min_x, max_x = min(all_x_coords), max(all_x_coords)
    min_y, max_y = min(all_y_coords), max(all_y_coords)
    # Рассчитать границы обрезки с запасом 10%
    padding_x = int((max_x - min_x) * padding_ratio)
    padding_y = int((max_y - min_y) * padding_ratio)
    crop_min_x = max(min_x - padding_x, 0)
    crop_max_x = min(max_x + padding_x, rotated_image.width)
    crop_min_y = max(min_y - padding_y, 0)
    crop_max_y = min(max_y + padding_y, rotated_image.height)
    # Обрезать изображение
    cropped_image = rotated_image.crop((crop_min_x, crop_min_y, crop_max_x, crop_max_y))
    # Смещение точек, линий и полигона
    shift_x = crop_min_x
    shift_y = crop_min_y
    shifted_points = [(x - shift_x, y - shift_y) for x, y in rotated_points]
    shifted_lines = [((x1 - shift_x, y1 - shift_y), (x2 - shift_x, y2 - shift_y), name) for (x1, y1), (x2, y2), name in rotated_lines]
    shifted_roi_coords = [(x - shift_x, y - shift_y) for x, y in rotated_roi_coords]
    return cropped_image, shifted_points, shifted_lines, shifted_roi_coords

def transform_to_parallelogram(quad):
    p1, p2, p3, p4 = quad
    v1 = p2 - p1
    v2 = p3 - p2
    p4_new = p3 - v1
    parallelogram = np.array([p1, p2, p3, p4_new])
    return parallelogram

def get_extreme_points(roi_coords, num_points=4):
    # Построение выпуклой оболочки
    hull = ConvexHull(roi_coords)
    hull_points = roi_coords[hull.vertices]
    # Найти все пары точек и их расстояния
    max_perimeter = 0
    best_points = None
    for combo in combinations(hull_points, num_points):
        perimeter = 0
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                perimeter += np.linalg.norm(combo[i] - combo[j])
        if perimeter > max_perimeter:
            max_perimeter = perimeter
            best_points = combo
    best_points = np.array(best_points)    
    # Вычисление центра масс для сортировки точек по кругу
    center = np.mean(best_points, axis=0)    
    def polar_angle(point):
        return np.arctan2(point[1] - center[1], point[0] - center[0])  
    
    # Сортировка точек по полярному углу
    
    # Сортировка точек по полярному углу
    sorted_points = sorted(best_points, key=polar_angle)
    # Преобразование в numpy массив
    sorted_points = np.array(sorted_points)
    
    
    # Нахождение правой нижней точки
    rightmost_bottom_point = max(sorted_points, key=lambda point: (point[0], -point[1]))
    
    # Найти индекс правой нижней точки в отсортированном списке
    start_index = np.where((sorted_points == rightmost_bottom_point).all(axis=1))[0][0]
    
    # Сдвиг отсортированного списка так, чтобы правая нижняя точка была первой
    best_points_sorted = np.concatenate((sorted_points[start_index:], sorted_points[:start_index]))

    return transform_to_parallelogram(np.array(sorted_points))

def transform_polygon_to_rect(image, roi_coords, output_size=(512, 512)):
    # Преобразование изображения в формат numpy
    image_np = np.array(image)    
    quad_coords = get_extreme_points(roi_coords)    
    # Определение целевого прямоугольника
    dst_points = np.float32([
        [0, 0],
        [output_size[0] - 1, 0],
        [output_size[0] - 1, output_size[1] - 1],
        [0, output_size[1] - 1]
    ])    
    # Преобразование изображения
    M = cv2.getPerspectiveTransform(np.float32(quad_coords), dst_points)
    warped_image_np = cv2.warpPerspective(image_np, M, output_size)    
    # Преобразование результата обратно в формат PIL
    warped_image = Image.fromarray(warped_image_np)    
    return warped_image



def prepare_data(protocol_path, exps_input):
    df = pd.read_excel(protocol_path)
    # Filter out rows where 'take_to_stat' is 'no' and log removed experiments
    removed_experiments = df[df['take_to_stat'] == 'no']['ID'].tolist()
    df = df[df['take_to_stat'] != 'no']
    df.reset_index(drop=True, inplace=True)    
    exps_input = parce_exps(exps_input)    
    # Find rows with the specified experiment numbers
    if exps_input == 'all':
        rows_to_process = list(range(len(df)))
    else:
        rows_to_process = df[df['ID'].isin(exps_input)].index.tolist()        
    return df, rows_to_process

def parce_exps(exps_input):
    exps_input = exps_input.strip().lower()
    if exps_input == 'all':
        return 'all'
    # Check if input is in the range format
    if ':' in exps_input:
        start, end = map(int, exps_input.split(':'))
        return list(range(start, end + 1))
    else:
        return [int(exp.strip()) for exp in exps_input.split(',')]

def remove_large_objects(ar, max_size):
    # Label connected components
    labeled = label(ar)
    for region in regionprops(labeled):
        if region.area > max_size:
            ar[labeled == region.label] = 0
    return ar

def distance(p1, p2):
    return round(np.linalg.norm(p1 - p2))

def sides_lengths(quad_coords):
    A, B, C, D = quad_coords
    AB = distance(A, B)
    BC = distance(B, C)
    return (AB, BC)

from PIL import Image, ImageDraw
exps_input = 'all'
protocol_path = "E:\iMAGES\protocol.xlsx"
df, rows_to_process = prepare_data(protocol_path, exps_input)

with tqdm(total=len(rows_to_process), desc="Processing images") as pbar:
    for idx, row_idx in enumerate(rows_to_process):
        file_path = df.iloc[row_idx]['filepath']
    
        base_name = splitext(basename(file_path))[0]
        experiment_date = basename(dirname(file_path))
        csv_path = join(dirname(file_path), f"{experiment_date}_{base_name}_comments.csv")
        
        if os.path.isfile(csv_path):
        
            image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_with_roi.png")
            
            denoised_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_denoised.png")
            
            roi_coords_path = join(dirname(file_path), f"{experiment_date}_{base_name}_roi_coords.csv")
            
            # Открытие изображения
            image = Image.open(image_path)
            denoised_image = Image.open(denoised_image_path)
            
            threshold_value = threshold_otsu(np.array(denoised_image))
            binary_image = np.array(denoised_image) > threshold_value
            binary_image = remove_small_objects(binary_image, min_size=20)
            binary_image = remove_large_objects(binary_image, max_size=200)
            bin_image = Image.fromarray(binary_image)
            
            # Чтение координат из файла
            roi_coords_df = pd.read_csv(roi_coords_path, delimiter=';')
            roi_coords = roi_coords_df[['x', 'y']].values
            
            # Загрузка точек и линий
            points, point_names, lines = load_comments(csv_path)
            
            # Поиск линии "radial axis"
            radial_axis = next(line for line in lines if line[2] == "radial axis")
            
            # Вычисление угла
            angle = -calculate_angle(radial_axis) + 90
            
            # Центр вращения
            rotation_center = radial_axis[0]
            
            # Расширение изображения
            expanded_image, expanded_center, shift_x, shift_y = expand_image(image, angle, rotation_center)
            bin_expanded_image, expanded_center, shift_x, shift_y = expand_image(bin_image, angle, rotation_center)
            
            # Смещение точек и линий в соответствии с новым центром
            shifted_points, shifted_lines = shift_annotations(points, lines, shift_x, shift_y)
            # Смещение точек полигона
            shifted_roi_coords = [(x + shift_x, y + shift_y) for x, y in roi_coords]
            
            # Поворот изображения
            rotated_image = rotate_image(expanded_image, -angle, expanded_center)
            bin_rotated_image = rotate_image(bin_expanded_image, -angle, expanded_center)
            
            # Поворот точек и линий
            rotated_points, rotated_lines = rotate_annotations(shifted_points, shifted_lines, angle, expanded_center)
            
            # Поворот полигона
            rotated_roi_coords = rotate_polygon(shifted_roi_coords, angle, expanded_center)
            
            # Использование функции для обрезки изображения и смещения аннотаций
            cropped_image, shifted_points, shifted_lines, shifted_roi_coords = crop_rotated_image_with_annotations(
                rotated_image, rotated_points, rotated_lines, rotated_roi_coords
            )
            bin_cropped_image, shifted_points, shifted_lines, shifted_roi_coords = crop_rotated_image_with_annotations(
                bin_rotated_image, rotated_points, rotated_lines, rotated_roi_coords
            )
            
            shifted_roi_coords = np.array(shifted_roi_coords, dtype=np.float32)
            quad_coords = get_extreme_points(shifted_roi_coords)
            
            rotated_output_path = join(dirname(file_path), f"{experiment_date}_{base_name}_roi_coords_rotated.csv")
            rotated_roi_df = pd.DataFrame(rotated_roi_coords, columns=['x', 'y'])
            rotated_roi_df[['x', 'y']].to_csv(rotated_output_path, index=False, sep=';')
            
            quad_output_path = join(dirname(file_path), f"{experiment_date}_{base_name}_quad_coords.csv")
            quad_df = pd.DataFrame(quad_coords, columns=['x', 'y'])
            quad_df[['x', 'y']].to_csv(quad_output_path, index=False, sep=';')
            
            # установка точки для тестирования трансформации
            # x, y = (quad_coords[0, 0], quad_coords[0, 1])
            # point_names.append('test')
            # shifted_points.append((x, y))            
            # draw = ImageDraw.Draw(bin_cropped_image)
            # radius = 20
            # draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(0, 255, 0))
            
             # Преобразование полигона в прямоугольник
            warped_image = transform_polygon_to_rect(bin_cropped_image, quad_coords, output_size=sides_lengths(quad_coords))
            
            # Отображение повернутого изображения с аннотациями
            fig, ax = show_image(cropped_image, title=f" {experiment_date} {base_name}", points=shifted_points, point_names=point_names, lines=shifted_lines)
            polygon = Polygon(quad_coords, closed=True, edgecolor='#F5EC4D', facecolor='none', linewidth=2, alpha=0.7)
            ax.add_patch(polygon)
            image_file_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_rotated_image.png")
            plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close(fig)
            
            # Отображение искаженного изображения
            fig, ax = show_image(warped_image)
            image_file_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_warped_image.png")
            plt.savefig(image_file_path, bbox_inches='tight', pad_inches=0, dpi=200)
            plt.close(fig)
        
        pbar.update(1)
#%% собираем гистограммы
#%matplotlib qt

import numpy as np
import os
from os.path import join, basename, dirname, splitext
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

def calculate_brightness_histogram(image, bin_size):
    # Преобразовать изображение в оттенки серого
    gray_image = image.convert('L')    
    # Преобразовать изображение в массив numpy
    gray_array = np.array(gray_image)/255    
    # Определить количество строк и столбцов в изображении
    height, width = gray_array.shape 
    # Инициализировать список для хранения средней яркости блоков
    average_brightness = []    
    # Вычислить среднюю яркость для каждого блока
    for i in range(0, height, bin_size):
        block = gray_array[i:i+bin_size, :]
        block_mean = np.sum(block)/(width/bin_size)# нормализуем количество в блоке по длине возможного количества
        average_brightness.append(block_mean)        
    return average_brightness

exps_input = 'all'
locations = ['SO', 'SR', 'SP_CA1']
age_groups = [5, 11, 15, 21]
protocol_path = "E:\\iMAGES\\protocol.xlsx"
df, rows_to_process = prepare_data(protocol_path, exps_input)

# Создаем словарь для хранения гистограмм по локациям и возрастам
data_by_location_and_age = {location: {age: [] for age in age_groups} for location in locations}

for location_in in locations:
    for Age_in in age_groups:
        
        # Фильтрация нужного возраста
        filtered_rows = [row_idx for row_idx in rows_to_process if df.iloc[row_idx]['Postnatal_Age'] == Age_in]
        
        # Фильтрация строк, содержащих location_in
        filtered_rows = [row_idx for row_idx in filtered_rows if df.iloc[row_idx]['location'] == location_in]
        
        with tqdm(total=len(filtered_rows), desc=f"Processing images for age {Age_in} and location {location_in}") as pbar:
            for idx, row_idx in enumerate(filtered_rows):
                file_path = df.iloc[row_idx]['filepath']
                
                location = df.iloc[row_idx]['location']   
                
                base_name = splitext(basename(file_path))[0]
                experiment_date = basename(dirname(file_path))
                csv_path = join(dirname(file_path), f"{experiment_date}_{base_name}_comments.csv")
                
                if os.path.isfile(csv_path):
                    warped_image_file = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_warped_image.png")
                    
                    warped_image = Image.open(warped_image_file)
                    
                    # Обработка изображений в зависимости от location_in
                    if location_in == 'SP_CA1':
                        # Установить строгую высоту изображения
                        warped_image = warped_image.resize((warped_image.width, 1000))
                    elif location_in == 'SR':
                        # Перевернуть изображение вверх ногами
                        warped_image = warped_image.transpose(Image.FLIP_TOP_BOTTOM)
                    
                    bin_size = 7
                    average_brightness = calculate_brightness_histogram(warped_image, bin_size)
                    
                    # Построить гистограмму
                    x = np.arange(len(average_brightness))
                    y = average_brightness
                    plt.close('all')
                    plt.style.use('dark_background')
                    plt.figure(figsize=(6, 10))
                    plt.plot(y, x)
                    plt.gca().invert_yaxis()
                    plt.title(f"{experiment_date}_{base_name}_{location}")
                    plt.ylabel('Histogram blocks (1 um)')
                    plt.xlabel('Average number of synapses')
                    
                    histogram_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_{location}_histogram.png")
                    plt.savefig(histogram_path)
                    
                    histogram_csv_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_{location}_histogram.csv")
                    
                    histdf = pd.DataFrame(average_brightness)
                    histdf.to_csv(histogram_csv_path, index=False)                    

                    plt.show()
                    
                    data_by_location_and_age[location_in][Age_in].append(average_brightness)
                    
                pbar.update(1)
plt.close('all')
#%%
# Функция для объединения и обработки гистограмм
def process_histograms(hist_list):
    # Определяем максимальную длину гистограммы
    max_length = max(len(hist) for hist in hist_list)
    
    # Создаем двумерный массив для хранения всех гистограмм с выравниванием по максимальной длине
    histogram_array = np.zeros((len(hist_list), max_length))
    
    for i, hist in enumerate(hist_list):
        histogram_array[i, :len(hist)] = hist
    
    # Вычислить среднее значение для каждой строки
    average_histogram = np.mean(histogram_array, axis=0)
    
    return average_histogram

# Обработка данных по всем локациям и возрастам
processed_data = {location: {} for location in locations}
for location in locations:
    for age in age_groups:
        
        processed_data[location][age] = process_histograms(data_by_location_and_age[location][age])
        
        # # Фильтрация нужного возраста
        # filtered_rows = [row_idx for row_idx in rows_to_process if df.iloc[row_idx]['Postnatal_Age'] == age]
        # for idx, row_idx in  enumerate(tqdm(filtered_rows, desc="Processing Histograms")):
        #     file_path = df.iloc[row_idx]['filepath']            
        #     location = df.iloc[row_idx]['location']   
            
        #     base_name = splitext(basename(file_path))[0]
        #     experiment_date = basename(dirname(file_path))
            
        #     histogram_csv_path = os.path.join(os.path.dirname(file_path), f"{experiment_date}_{base_name}_{location}_histogram.csv")
        #     if os.path.exists(histogram_csv_path):
        #         average_brightness = pd.read_csv(histogram_csv_path).to_numpy()                
        #         processed_data[location][age] = average_brightness

# Найти максимальное значение всех данных
locations_to_plot = ['SO', 'SP_CA1', 'SR']
max_val = 0
for location in locations_to_plot:
    for age in age_groups:
        max_val = max(max_val, max(processed_data[location][age]))



#%% объединяем картинки подсчета радиальной оси

def combine_images(image_paths, output_path):
    images = [Image.open(img_path) for img_path in image_paths]
    widths, heights = zip(*(img.size for img in images))

    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    combined_image.save(output_path)

exps_input = 'all'
protocol_path = "E:\\iMAGES\\protocol.xlsx"
output_folder = "E:\\iMAGES\\radial axis results"
os.makedirs(output_folder, exist_ok=True)

df, rows_to_process = prepare_data(protocol_path, exps_input)

for idx, row_idx in enumerate(tqdm(rows_to_process, desc="Processing Images")):
    file_path = df.iloc[row_idx]['filepath']
    location = df.iloc[row_idx]['location']   
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    
    rotated_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_rotated_image.png")
    warped_image_file = join(dirname(file_path), f"{experiment_date}_{base_name}_warped_image.png")
    histogram_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{location}_histogram.png")
    
    if os.path.exists(rotated_image_path) and os.path.exists(warped_image_file) and os.path.exists(histogram_path):
        output_image_path = join(output_folder, f"{experiment_date}_{base_name}_combined.png")
        combine_images([rotated_image_path, warped_image_file, histogram_path], output_image_path)

#%% Создание фигуры и сабплотов
plt.style.use('default')

fig, axs = plt.subplots(3, 1, figsize=(6, 12), dpi=80)

# Названия локаций
locations_to_plot = ['SO', 'SP_CA1', 'SR']
location_titles = ['Oriens', 'Pyramidale', 'Radiatum']

# Построение данных для каждой локации
for i, location in enumerate(locations_to_plot):
    for age in age_groups:
        
        data = processed_data[location][age]
        
        x = np.arange(len(data))
        y = data
        
        # Повернуть график
        axs[i].plot(y, x, label=f'P{age}')
    
    axs[i].set_title(f'{location_titles[i]}')
    axs[i].set_xlabel('Average number of synapses')
    axs[i].set_ylabel('Histogram blocks (1 um)')
    
    # Устанавливаем лимиты осей
    axs[i].set_xlim([0, max_val])
    # axs[i].set_ylim([0, len(data)])
    
    if location == 'SO':
        # axs[i].invert_yaxis()  # Переворачиваем ось Y
        axs[i].legend()
    if location == 'SR':
        axs[i].invert_yaxis()  # Переворачиваем ось Y
        
    # Убираем рамки (box) для осей
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['left'].set_visible(False)
    axs[i].spines['bottom'].set_visible(False)
    
# Отображение графиков
plt.tight_layout()

# Сохранение графиков
output_path = 'E:\\iMAGES\\structure_histograms.png'
plt.savefig(output_path)

# Показ графиков
plt.show()
#%% сшиваем по краям

import matplotlib.pyplot as plt
import numpy as np

def stitching(arr1, arr2, window_size = 20):
    
    # Длина нового массива
    new_length = len(arr1) + len(arr2) - window_size
    
    # Новый массив
    stitched_array = np.zeros(new_length)
    
    # Копируем первую часть первого массива
    stitched_array[:len(arr1)-window_size] = arr1[:-window_size]
    
    # Сшивание с плавным переходом
    for i in range(window_size):
        weight = i / window_size
        stitched_array[len(arr1)-window_size+i] = (1 - weight) * arr1[-window_size+i] + weight * arr2[i]
    
    # Копируем оставшуюся часть второго массива
    stitched_array[len(arr1):] = arr2[window_size:]
    return stitched_array

def create_overlap(data, locations, age_groups, window_size):
    
    # Создаем копию данных, чтобы не изменять оригинальные данные
    modified_data = {loc: {age: data[loc][age].copy() for age in age_groups} for loc in locations}

    for age in age_groups:
        # Extract the arrays for SO and SP_CA1
        so_array = data[locations[0]][age]
        sp_ca1_array = data[locations[1]][age]
        sr_array = data[locations[2]][age]
        
        # Сшиваем SO и SP
        stitched_array = double_stitching(sp_ca1_array, so_array, window_size)        
        div_index = len(sp_ca1_array)
        new_sp_ca1_array = stitched_array[:div_index]
        new_so_array = stitched_array[div_index:]        
        
        # Сшиваем SP и SR
        stitched_array = double_stitching(np.flip(sr_array), new_sp_ca1_array, window_size)        
        div_index = len(sr_array)
        new_sr_array = np.flip(stitched_array[:div_index])
        new_sp_ca1_array = stitched_array[div_index:]
        
        modified_data[locations[0]][age] = new_so_array
        modified_data[locations[1]][age] = new_sp_ca1_array
        modified_data[locations[2]][age] = new_sr_array
        
    return modified_data

def plot_before_after(data_before, data_after, locations, age):
    fig, axes = plt.subplots(len(locations), 2, figsize=(12, 8))
    
    for i, loc in enumerate(locations):
        ax_before = axes[i, 0]
        ax_after = axes[i, 1]
        
        ax_before.plot(data_before[loc][age], label=f'{loc} Before')
        ax_before.set_title(f'{loc} Before')
        ax_before.legend()
        
        ax_after.plot(data_after[loc][age], label=f'{loc} After')
        ax_after.set_title(f'{loc} After')
        ax_after.legend()
    
    plt.tight_layout()
    plt.show()

locations_to_plot = ['SO', 'SP_CA1', 'SR']
age_groups = [5, 11, 15, 21]
window_size = round(16.5)
half_window = 1+(window_size//2)

# Store the original data for comparison
original_data = {loc: {age: data.copy() for age, data in processed_data[loc].items()} for loc in locations_to_plot}

# Apply the function to the data
modified_data = create_overlap(processed_data, locations_to_plot, age_groups, window_size)

# Plot before and after for age group 5
plot_before_after(original_data, modified_data, locations_to_plot, 21)

#%% Создание фигуры и сабплотов (полностью склеенный)

import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
fig, axs = plt.subplots(1, 1, figsize=(6, 12), dpi=80)

# Названия локаций
locations_to_plot = ['SO', 'SP_CA1', 'SR']
location_titles = ['Oriens', 'Pyramidale', 'Radiatum']

# Цветовая палитра для разных возрастных групп
colors = plt.get_cmap('tab10')

# Объединенные данные
combined_data = {age: [] for age in age_groups}

# Индексы переходных зон
transition_indices = {age: [] for age in age_groups}

# Объединение данных из разных локаций
for location in locations_to_plot:
    for age in age_groups:
        data = modified_data[location][age]
        if location in ['SO', 'SP_CA1']:
            data = data[::-1]  # Переворачиваем данные
        combined_data[age].extend(data)
        transition_indices[age].append(len(combined_data[age])+half_window)

# Построение данных для каждого возраста
for j, age in enumerate(age_groups):
    
    data = combined_data[age]
    
    x = np.arange(len(data))
    y = data
    
    # Определение индексов для точек window_size
    split_index_start = half_window if len(data) > half_window else len(data)
    split_index_end = len(data) - half_window if len(data) > half_window else 0

    # Отрисовка основной линии и штриховых линий в зонах перехода
    prev_transition = 0
    for transition_index in transition_indices[age]:
        # Основная линия до переходной зоны
        axs.plot(y[prev_transition:transition_index - half_window], x[prev_transition:transition_index - half_window], label=f'P{age}' if prev_transition == 0 else "", color=colors(j))
        
        # Штриховая линия в переходной зоне
        if transition_index - half_window > 0:
            axs.plot(y[transition_index - half_window-1:transition_index+1], x[transition_index - half_window-1:transition_index+1], linestyle=':', color=colors(j))
        if transition_index < len(data) - half_window:
            axs.plot(y[transition_index-1:transition_index + half_window+1], x[transition_index-1:transition_index + half_window+1], linestyle=':', color=colors(j))
        
        prev_transition = transition_index + half_window

    # Основная линия после последней переходной зоны
    axs.plot(y[prev_transition:], x[prev_transition:], color=colors(j))

# Получаем среднее значение для Y, чтобы расположить текст в середине по оси Y
y_middle = (max(x) - min(x)) / 2
x_middle = max_val

# Размещение текста в середине оси Y для каждой локации
for i, location in enumerate(locations_to_plot):
    axs.text(x_middle, y_middle + (i-1) * len(data) // len(locations_to_plot), location_titles[i], 
             fontsize=12, va='center', ha='left', rotation='horizontal')

axs.set_xlabel('Average number of synapses')
axs.set_ylabel('Histogram blocks (1 um)')

# Устанавливаем лимиты осей
axs.set_xlim([0, max_val*1.1])
axs.invert_yaxis()
# axs.set_ylim([0, len(data)])

# Убираем рамки (box) для осей
axs.spines['top'].set_visible(False)
axs.spines['right'].set_visible(False)
axs.spines['left'].set_visible(False)
axs.spines['bottom'].set_visible(False)

axs.legend()

# Точное подгонка расстояния между графиками
plt.subplots_adjust(hspace=0)

# Отображение графиков
# plt.tight_layout()

# Сохранение графиков
output_path = 'E:\\iMAGES\\structure_histograms(cross-stiched).png'
plt.savefig(output_path)

# Показ графиков
plt.show()


#%% пример сшивания

import numpy as np
import matplotlib.pyplot as plt

# Исходные массивы
arr1 = np.array(range(100))
arr2 = np.array(range(100, 200, 2))

def stitching(arr1, arr2, window_size = 20):
    
    # Длина нового массива
    new_length = len(arr1) + len(arr2) - window_size
    
    # Новый массив
    stitched_array = np.zeros(new_length)
    
    # Копируем первую часть первого массива
    stitched_array[:len(arr1)-window_size] = arr1[:-window_size]
    
    # Сшивание с плавным переходом
    for i in range(window_size):
        weight = i / window_size
        stitched_array[len(arr1)-window_size+i] = (1 - weight) * arr1[-window_size+i] + weight * arr2[i]
    
    # Копируем оставшуюся часть второго массива
    stitched_array[len(arr1):] = arr2[window_size:]
    
    
    return stitched_array

window_size = 20
stitched_array = stitching(arr1, arr2, window_size)

new_length = len(arr1) + len(arr2) - window_size

fig, axs = plt.subplots(1, 1)
# Визуализация результата
plt.plot(stitched_array, label='Stitched Array')
plt.plot(range(len(arr1)), arr1, label='Array 1', linestyle='--')
plt.plot(range(len(arr1)-window_size, new_length), arr2, label='Array 2', linestyle='--')
plt.legend()

x_values = [arr1[-1]-window_size, arr1[-1], arr1[-1]+window_size]

for x in x_values:
    plt.axhline(y=x, color='r', linestyle='--')
plt.show()
#%% двойное сшивание для перекрытия
import numpy as np
import matplotlib.pyplot as plt

# Исходные массивы
arr1 = np.array(range(100))
arr2 = np.array(range(100, 200, 2))

def stitching(arr1, arr2, window_size):
    
    # Длина нового массива
    new_length = len(arr1) + len(arr2) - window_size
    
    # Новый массив
    stitched_array = np.zeros(new_length)
    
    # Копируем первую часть первого массива
    stitched_array[:len(arr1)-window_size] = arr1[:-window_size]
    
    # Сшивание с плавным переходом
    for i in range(window_size):
        weight = i / window_size
        stitched_array[len(arr1)-window_size+i] = (1 - weight) * arr1[-window_size+i] + weight * arr2[i]
    
    # Копируем оставшуюся часть второго массива
    stitched_array[len(arr1):] = arr2[window_size:]    
    
    return stitched_array

def double_stitching(arr1, arr2, window_size):
    # Первый этап сшивания
    stitched_array = stitching(arr1, arr2, window_size)
    
    # Точка сшивания
    stitch_point = len(arr1) - window_size
    
    half_window = window_size//2
    # Разделение сшитого массива на две части
    left_part = stitched_array[:stitch_point + half_window]
    right_part = stitched_array[stitch_point + half_window:]
    
    # Уменьшение окна в два раза
    new_window_size = half_window
    
    # Повторное сшивание уменьшенных частей
    final_stitched_array = stitching(left_part, right_part, new_window_size)
    
    return final_stitched_array

window_size = 10
double_stitched_array = double_stitching(arr1, arr2, window_size)

fig, axs = plt.subplots(1, 1)
# Визуализация результата
plt.plot(double_stitched_array, label='Double Stitched Array')
plt.plot(range(len(arr1)), arr1, label='Array 1', linestyle='--')
plt.plot(range(len(arr1)-window_size, len(arr1) + len(arr2) - window_size), arr2, label='Array 2', linestyle='--')
plt.legend()

x_values = [arr1[-1]-window_size, arr1[-1], arr1[-1]+window_size]
for x in x_values:
    plt.axhline(y=x, color='r', linestyle='--')
plt.show()
#%% определяем перекрытие

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from os.path import join, dirname, basename, splitext, isfile
import matplotlib.pyplot as plt

# Функция для создания маски из координат полигона
def create_mask(coords, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [coords], 1)
    return mask

# Функция для вычисления перекрытия вдоль ребер параллелограмма
def compute_edge_overlap(mask_diff, quad_coords, thickness=100):
    edges = [(quad_coords[i], quad_coords[(i + 1) % 4]) for i in range(4)]
    stats = []

    for edge in edges:
        p1, p2 = edge
        length = int(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        x_vals = np.linspace(p1[0], p2[0], length).astype(int)
        y_vals = np.linspace(p1[1], p2[1], length).astype(int)
        
        edge_strip = []

        for i in range(-thickness//2, thickness//2 + 1):
            # Вектор нормали к ребру
            dx = y_vals - y_vals.mean()
            dy = x_vals - x_vals.mean()
            norm = np.sqrt(dx**2 + dy**2)
            dx = (dx / norm * i).astype(int)
            dy = (dy / norm * i).astype(int)
            
            x_offset = x_vals + dx
            y_offset = y_vals + dy

            # Ограничиваем координаты внутри изображения
            x_offset = np.clip(x_offset, 0, mask_diff.shape[1] - 1)
            y_offset = np.clip(y_offset, 0, mask_diff.shape[0] - 1)

            overlap = mask_diff[y_offset, x_offset]
            edge_strip.append(overlap)
        
        stats.append(np.array(edge_strip))
    
    return stats

# Основной цикл обработки
def process_images(protocol_path, locations, age_groups, exps_input='all'):
    df, rows_to_process = prepare_data(protocol_path, exps_input)
    data_by_location_and_age = {location: {age: [] for age in age_groups} for location in locations}

    for location_in in locations:
        for Age_in in age_groups:
            filtered_rows = [row_idx for row_idx in rows_to_process if df.iloc[row_idx]['Postnatal_Age'] == Age_in]
            filtered_rows = [row_idx for row_idx in filtered_rows if df.iloc[row_idx]['location'] == location_in]
            
            for idx, row_idx in enumerate(tqdm(filtered_rows, desc="Processing Images ")):
                file_path = df.iloc[row_idx]['filepath']
                base_name = splitext(basename(file_path))[0]
                experiment_date = basename(dirname(file_path))
                
                rotated_output_path = join(dirname(file_path), f"{experiment_date}_{base_name}_roi_coords_rotated.csv")
                
                if not isfile(rotated_output_path):
                    continue
                
                rotated_roi_df = pd.read_csv(rotated_output_path, delimiter=';')
                rotated_roi_coords = rotated_roi_df[['x', 'y']].values.astype(np.int32)

                quad_output_path = join(dirname(file_path), f"{experiment_date}_{base_name}_quad_coords.csv")
                quad_df = pd.read_csv(quad_output_path, delimiter=';')
                quad_coords = quad_df[['x', 'y']].values.astype(np.int32)
                
                # Вычисление размеров изображения
                all_coords = np.vstack((rotated_roi_coords, quad_coords))
                x_min, y_min = np.min(all_coords, axis=0)
                x_max, y_max = np.max(all_coords, axis=0)
                shape = (y_max - y_min + 1, x_max - x_min + 1)
                
                # Сдвиг координат в положительные области
                rotated_roi_coords_shifted = rotated_roi_coords - [x_min, y_min]
                quad_coords_shifted = quad_coords - [x_min, y_min]
                
                # Создание масок
                mask_roi = create_mask(rotated_roi_coords_shifted, shape)
                mask_quad = create_mask(quad_coords_shifted, shape)
                
                # Логическая разница
                mask_diff = cv2.bitwise_xor(mask_roi, mask_quad)
                
                # Вычисление статистики перекрытия вдоль ребер параллелограмма
                stats = compute_edge_overlap(mask_diff, quad_coords_shifted)
                
                data_by_location_and_age[location_in][Age_in].append(stats)
    
    return data_by_location_and_age


# Пример использования
protocol_path = "E:\\iMAGES\\protocol.xlsx"
locations = ['SO', 'SR', 'SP_CA1']
age_groups = [5, 11, 15, 21]
exps_input = 'all'

data_by_location_and_age = process_images(protocol_path, locations, age_groups, exps_input)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#%% перекрытие между слоями
def find_spike_edges(trace):
    # Нормализуем трейс
    trace_normalized = trace - np.min(trace)
    trace_normalized /= np.max(trace_normalized)
    
    # Находим пики
    peaks, _ = find_peaks(trace_normalized)
    
    if len(peaks) == 0:
        return None, None
    
    # Находим индекс максимального пика
    max_peak_idx = peaks[np.argmax(trace_normalized[peaks])]
    
    # Определяем возвращение на базовую линию (после максимального пика)
    return_to_baseline = None
    for i in range(max_peak_idx, len(trace_normalized)):
        if trace_normalized[i] < 0.1:  # Условие для возвращения к базовой линии
            return_to_baseline = i
            break
    
    if return_to_baseline is None:
        return_to_baseline = len(trace_normalized) - 1
    
    return max_peak_idx, return_to_baseline

def smooth_trace(trace, window_size=5):
    # Проверка, чтобы размер окна был положительным и не превышал длину трейса
    if window_size < 1:
        raise ValueError("Window size must be at least 1.")
    if window_size > len(trace):
        raise ValueError("Window size must not be larger than the trace length.")
    
    # Вычисление скользящего среднего с учетом сдвига
    smoothed_trace = np.convolve(trace, np.ones(window_size)/window_size, mode='valid')
    
    # Добавление начальных и конечных значений, чтобы выровнять длину
    start_padding = np.full(window_size//2, smoothed_trace[0])
    end_padding = np.full(window_size - window_size//2 - 1, smoothed_trace[-1])
    smoothed_trace = np.concatenate((start_padding, smoothed_trace, end_padding))
    
    return smoothed_trace

def detrend_data(trace):
    x = np.arange(len(trace))
    coeffs = np.polyfit(x, trace, 1)
    trend = np.polyval(coeffs, x)
    detrended_trace = trace - trend    
    return detrended_trace

#% Функция для вывода среднего трека перекрытия для одной локации и всех возрастов
def structure_overlap(data_by_location_and_age, location, age_groups, show_plot = False):
    if show_plot:
        plt.close('all')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    
    spike_differences = {}

    for edge_index in range(4):
        all_age_traces = []
        
        for age in age_groups:
            edges_data = data_by_location_and_age[location][age]
            if len(edges_data) == 0:
                print(f"No data available for location {location} and age {age}")
                continue

            # Усреднение данных для каждого ребра
            example_edge_data = [edge[edge_index] for edge in edges_data]

            # Усреднение каждого двумерного массива вдоль второй координаты
            averaged_traces = [np.mean(edge, axis=1) for edge in example_edge_data]

            # Усреднение полученных одномерных массивов
            avg_example_edge_data = np.abs(detrend_data(np.mean(averaged_traces, axis=0)))
            all_age_traces.append(avg_example_edge_data)
            
            if show_plot:
                axes[edge_index].plot(avg_example_edge_data, label=f'Age {age}')
            
        
        if edge_index in [1, 3]:
            # Рассчитываем средний абсолютный трейс между возрастами
            avg_trace_across_ages = smooth_trace(np.mean(all_age_traces, axis=0),window_size=3)
            
            
            # Изображаем средний трейс черным цветом
            if show_plot:
                axes[edge_index].plot(avg_trace_across_ages, color='k', linestyle='-', linewidth=2, label='Average Trace')
            
            # Определяем точки перегиба
            onset_spike, return_to_baseline = find_spike_edges(avg_trace_across_ages)
            
            if onset_spike is not None and return_to_baseline is not None:
                # Изображаем точки перегиба на оси
                if show_plot:
                    axes[edge_index].axvline(onset_spike, color='r', linestyle='--', label='Spike Onset')
                    axes[edge_index].axvline(return_to_baseline, color='g', linestyle='--', label='Return to Baseline')
                
                # Подсчитываем разницу по оси x
                spike_difference = return_to_baseline - onset_spike
                spike_differences[edge_index + 1] = spike_difference
                
                if show_plot:
                    # Добавляем численную разницу на график
                    axes[edge_index].text(0.5, 0.9, f'Difference: {spike_difference}', transform=axes[edge_index].transAxes, 
                                      fontsize=12, verticalalignment='top')
                
        if show_plot:
            axes[edge_index].set_title(f'Edge {edge_index + 1}')
            axes[edge_index].set_xlabel('Position along the edge')
            axes[edge_index].set_ylabel('Overlap')
            axes[edge_index].legend()
    
    if show_plot:
        plt.suptitle(f'Average Overlap for All Edges at Location {location}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    
    return spike_differences

# определяем перекрытие между структурами
locations = ['SO', 'SP_CA1', 'SR']
plot_ages = [5, 11, 15, 21]
all_mean_diff = [];
for plot_location in locations:
    spike_differences = structure_overlap(data_by_location_and_age, plot_location, plot_ages)

    # Вывод разниц в числовом виде
    print("Spike differences for edges 2 and 4:")
    for edge, diff in spike_differences.items():
        print(f'Edge {edge}: {diff} units')
    
    mean_difference = np.mean(list(spike_differences.values()))
    print(f'average Diff {mean_difference}')
    
    all_mean_diff.append(mean_difference)

SO_SP = all_mean_diff[0]+all_mean_diff[1]
SP_SR = all_mean_diff[1]+all_mean_diff[2]
Average_Structure_overlap = np.mean([SP_SR, SO_SP])

print(f'Radiatum-Oriens overlap {SO_SP}')
print(f'Pyramidale-Rradiatum overlap {SP_SR}')
print(f'Average Structure overlap {Average_Structure_overlap}')
#%% lif file
from readlif.reader import LifFile
import numpy as np
import matplotlib.pyplot as plt

# Путь к файлу .lif
file_path = r"C:\Users\ta3ma\Downloads\wetransfer_test_emx1-cre_full-shcdh13_p13_2024-08-02_1204\Test_Emx1-Cre_Full-shCdh13_P13\Mouse#1_Female.lif"

target_ch = 2
dapi_ch = 0

slice_start=2
slice_end=6
    
lif = LifFile(file_path)

img_list = [i for i in lif.get_iter_image()]

def collect_all_frames(im_in, ch):
    
    z_list = [i for i in im_in.get_iter_z(t=0, c=0)]
    z_n = len(z_list)
        
    channel_list = [i for i in im_in.get_iter_c(t=0, z=0)]
    ch_n = len(channel_list)
    
    def remove_shift(lst, ch):
        shift = ((ch_n-1) * ch - 1) % len(lst)
        reverse_shift = len(lst) - shift
        return lst[reverse_shift:] + lst[:reverse_shift]
    
    ch = (ch_n - ch)
    
    frames_out = []
    for z_real in list(range(z_n)):    
        z=(z_real*ch_n)%z_n       
        c=(z%ch_n+ch)%ch_n    
        frames_out.append(np.array(im_in.get_frame(z = z, c = c)))
    frames_out = np.array(remove_shift(frames_out, ch))
    
    return frames_out

for image in img_list:
    scale_factor = (16 - image.bit_depth[0]) ** 2  
    
    
    slide = list(range(slice_start-1, slice_end))
    # slide = [0, 4, 8]
    channel_list = [i for i in image.get_iter_c(t=0, z=0)]
    ch_n = len(channel_list)
        
    frames_1 = collect_all_frames(image, target_ch)
    frames_2 = collect_all_frames(image, dapi_ch)
        
    sample_slice_1 = np.max(frames_1[slide,:,:], axis=0)
    sample_slice_3 = np.max(frames_2[slide,:,:], axis=0)
    
    combined_image = zeros((*sample_slice_1.shape, 3), dtype='uint8')
    sample_slice_1_normalized = (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1)) * 255
    sample_slice_3_normalized = (sample_slice_3 - np.min(sample_slice_3)) / (np.max(sample_slice_3) - np.min(sample_slice_3)) * 255

    combined_image[:, :, 0] = sample_slice_1_normalized  # synaptotagmin channel
    combined_image[:, :, 2] = sample_slice_3_normalized  # cell-label dapi channel
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(combined_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
#%%



target_ch = 3
frames_out = collect_all_frames(image, target_ch)

for test_image in frames_out:
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 1, 1)
    plt.imshow(test_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
#%%

initial_location = ''

if not initial_location:
    print('empty')
else:
    print('not empty')
    
#%%
import pandas as pd

# Чтение Excel-файла с автоматически добавленными суффиксами
coords_path = r"E:\Tile Photos\240812 AG\Experiment-1538_results\Experiment-1538_0_locations.xlsx"
coords_df = pd.read_excel(coords_path)

# Функция для переименования колонок, которые содержат суффиксы вида .1, .2
def rename_column_names(columns):
    new_columns = []
    
    for col in columns:
        if '.' in col:  # Если есть точка и номер
            # Разделяем название колонки на части
            base_name, suffix = col.split('.')
            # Переносим номер (suffix) перед '_x' или '_y'
            if '_x' in base_name:
                new_col = base_name.replace('_x', f'_{suffix}_x')
            elif '_y' in base_name:
                new_col = base_name.replace('_y', f'_{suffix}_y')
            else:
                new_col = base_name  # На случай, если нет "_x" или "_y"
        else:
            new_col = col  # Оставляем без изменений, если нет точки
        
        new_columns.append(new_col)
    
    return new_columns

# Переименовываем колонки с дубликатами
coords_df.columns = rename_column_names(coords_df.columns)

# Выводим обновленные имена колонок для проверки
print(coords_df.columns)

# Работа с данными (парсинг координат)
for col_x in coords_df.columns[::2]:  # Итерация по парам колонок x, y
    col_y = col_x.replace('_x', '_y')  # Поиск соответствующей y-колонки
    location_name = col_x.rsplit('_', 2)[0]  # Извлечение имени области без индекса и координаты
    coords = coords_df[[col_x, col_y]].values
    print(f"Location: {location_name}, Coordinates: {coords}")
