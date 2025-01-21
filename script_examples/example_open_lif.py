#%%
from readlif.reader import LifFile
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\ta3ma\Downloads\wetransfer_test_emx1-cre_full-shcdh13_p13_2024-08-02_1204\Test_Emx1-Cre_Full-shCdh13_P13\Mouse#1_Female.lif"
lif = LifFile(file_path)    

def collect_lif_frames(im_in, ch, z_n, ch_n):    
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

target_ch = 0

img_list = [i for i in lif.get_iter_image()]
combined_image_s = []
for im_index, im_in in enumerate(img_list):      
    
    z_list = [i for i in im_in.get_iter_z(t=0, c=0)]
    z_n = len(z_list)# число глубин 
        
    channel_list = [i for i in im_in.get_iter_c(t=0, z=0)]
    ch_n = len(channel_list)# число каналов
    
    frames = collect_lif_frames(im_in, target_ch, z_n, ch_n)# массив фреймов выбранного канала target_ch
    
#%%
import numpy as np
from PIL import Image

def process_tif_image(file_path):
    img = Image.open(file_path)
    
    img_stack = []        
    # Проходим по каждому слою в многослойном изображении
    for i in range(img.n_frames):
        img.seek(i)
        
        # Преобразуем слой в массив и нормализуем в 8-битный диапазон
        layer = np.array(img, dtype=np.uint16)
        layer_max = layer.max()
        
        # Проверка на максимальное значение, чтобы избежать деления на ноль
        if layer_max > 0:
            layer_normalized = (layer * (255.0 / layer_max)).astype(np.uint8)
        else:
            layer_normalized = layer.astype(np.uint8)  # Если max=0, просто переводим в uint8
            
        # Дублируем слой по трем каналам для RGB
        rgb_layer = np.stack([layer_normalized] * 3, axis=-1)
        img_stack.append(rgb_layer)
    
    # Объединяем все слои в одно RGB-изображение, используя максимумы по каждому каналу
    combined_array = np.max(np.stack(img_stack), axis=0)
    
    # Преобразуем в PIL-изображение и гарантируем режим RGB
    combined_img = Image.fromarray(combined_array, mode='RGB')
    if combined_img.mode != 'RGB':
        combined_img = combined_img.convert('RGB')
    
    return combined_img

# Пример использования
file_path = r"C:\Users\ta3ma\Downloads\wetransfer_test_emx1-cre_full-shcdh13_p13_2024-08-02_1204\Test_Emx1-Cre_Full-shCdh13_P13\Mouse#1_Female_channel0_image1.tif"
processed_tif_img = process_tif_image(file_path)
processed_tif_img.show()


#%%

from czifile import CziFile
import numpy as np
from matplotlib import cm
from PIL import Image, ImageTk, ImageDraw, ImageFont

def normalize_channel(channel_data):
    """Нормирует данные канала в диапазон от 0 до 255"""
    channel_min = channel_data.min()
    channel_max = channel_data.max()
    if channel_max > channel_min:
        normalized_data = (channel_data - channel_min) / (channel_max - channel_min) * 255
    else:
        normalized_data = np.zeros_like(channel_data)
    return normalized_data.astype(np.uint8)

def create_combined_image(image_data, channels, slices, scale, max_channel, max_slice):
    """Создает RGB-изображение из заданных каналов и слайдов с опциональным масштабированием"""
    
    # Фильтруем только валидные каналы и срезы
    valid_channels = [ch for ch in channels if ch < max_channel]  # Игнорируем каналы, которые out of bounds
    valid_slices = [sl for sl in slices if sl < max_slice]        # Игнорируем срезы, которые out of bounds

    sample_slices = []
    
    # Получаем данные по каждому каналу через максимумы по валидным срезам
    for ch in valid_channels:
        sample_slice = np.max(image_data[0, 0, ch, valid_slices, :, :, 0], axis=0)
        sample_slices.append(sample_slice)
    
    # Если scale не равен 1, уменьшаем изображение
    if scale != 1.0:
        sample_slices = [zoom(slice_data, (scale, scale), order=1) for slice_data in sample_slices]
    
    # Инициализируем пустое RGB изображение
    combined_image = np.zeros((*sample_slices[0].shape, 3), dtype='uint8')
    
    # Выбираем цвета для каналов (RGB и дополнительные)
    colormap = cm.get_cmap('rainbow', len(valid_channels))  # Используем колорбар для распределения цветов по каналам
    
    # Проходим по каналам и добавляем их в RGB-изображение
    for i, sample_slice in enumerate(sample_slices):
        normalized_slice = normalize_channel(sample_slice)
        color = np.array(colormap(i)[:3]) * 255  # Получаем RGB цвет из колорбара
        for j in range(3):  # RGB компоненты
            combined_image[:, :, j] += (normalized_slice * color[j] / 255).astype(np.uint8)

    return combined_image

def process_czi_image(file_path, channels=None, slices=None, scale=1.0):
    """CZI image processing with optional arguments for selecting channels, slides and scale"""
    # Open the CZI file and get the data
    with CziFile(file_path) as czi:
        image_data = czi.asarray()
        if len(image_data.shape) == 8:  # If there are tiles
            image_data = image_data[:, :, 0, :, :, :, :, :]
        max_slice = image_data.shape[3]
        max_channel = image_data.shape[2]
    
    # If channels or slides are not specified, use all
    if channels is None:
        channels = list(range(max_channel))  # Use all channels
    if slices is None:
        slices = list(range(max_slice))  # Use all slides
    
    # Call a function to create a combined image from selected channels and slides
    combined_image = create_combined_image(image_data, channels, slices, scale, max_channel, max_slice)
    img = Image.fromarray(combined_image)
    
    return img

file_path = r"E:\iMAGES\P21.2 2 slide 2 slice6\Experiment-1000.czi"
processed_czi_img = process_czi_image(file_path)
processed_czi_img.show()
#%%
from readlif.reader import LifFile
from PIL import Image, ImageOps

def process_lif_image(file_path, scale=0.3):
    lif = LifFile(file_path)
    
    # Извлекаем, конвертируем в RGB, масштабируем и добавляем рамку к каждому изображению
    border_color = (128, 40, 128)  # цвет рамки
    border_size = 5  # Толщина рамки
    pil_images = []
    for img in lif.get_iter_image():
        plane = img.get_plane().convert("RGB")
        
        # Масштабируем изображение
        new_size = (int(plane.width * scale), int(plane.height * scale))
        scaled_image = plane.resize(new_size, Image.LANCZOS)
        
        # Добавляем зеленую рамку
        bordered_image = ImageOps.expand(scaled_image, border=border_size, fill=border_color)
        pil_images.append(bordered_image)

    # Определяем размеры для коллажа
    rows = int(len(pil_images) ** 0.5)
    cols = (len(pil_images) + rows - 1) // rows

    # Получаем размер каждого изображения с учетом рамки
    width, height = pil_images[0].size
    collage_width = cols * width
    collage_height = rows * height

    # Создаем пустой холст для коллажа
    collage = Image.new('RGB', (collage_width, collage_height))

    # Размещаем каждое изображение с рамкой в коллаже
    for i, img in enumerate(pil_images):
        x = (i % cols) * width
        y = (i // cols) * height
        collage.paste(img, (x, y))

    return collage

# Указываем путь к файлу .lif
file_path = r"C:\Users\ta3ma\Documents\Synapto_Catcher\data_examples\Mouse#1_Female.lif"
processed_lif_img = process_lif_image(file_path)
processed_lif_img.show()
