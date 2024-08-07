from os.path import splitext, basename, dirname, join, exists
from os import makedirs, path
from pandas import read_csv, DataFrame, concat, read_excel
from numpy import zeros, min as np_min, max as np_max, array, arange, meshgrid, vstack, histogram, finfo, log, argmax, asarray, mean, median
from matplotlib.pyplot import subplots, show, close, savefig, draw, imsave
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_isodata, threshold_mean, threshold_minimum
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from matplotlib.path import Path
from tqdm import tqdm
from scipy.fftpack import fft2, ifft2, fftshift
import numpy as np
import matplotlib.pyplot as plt
from czifile import CziFile
from aicspylibczi import CziFile as aicCzi
import cv2
import xml.etree.ElementTree as ET
import time
import pickle
from readlif.reader import LifFile

def extract_scaling_distances_from_czi(filename):
    # Open the .czi file
    czi = aicCzi(filename)
    
    # Get the metadata
    metadata_xml = czi.meta
    metadata_str = ET.tostring(metadata_xml, encoding='utf-8', method='xml')
    
    # Parsing the metadata string
    root = ET.fromstring(metadata_str)
    
    distances = {"X": None, "Y": None, "Z": None}
    
    # Looking for Scaling - Items - Distance
    for scaling in root.findall(".//Scaling/Items/Distance"):
        if 'Id' in scaling.attrib:
            id_value = scaling.attrib['Id']
            value = scaling.find('Value').text
            if id_value == 'X':
                distances["X"] = float(value)
            elif id_value == 'Y':
                distances["Y"] = float(value)
            elif id_value == 'Z':
                distances["Z"] = float(value)
    
    return distances

def filetype_checking(file_path):
    file_extension = splitext(file_path)[1].lower()  
    if file_extension == '.czi':
        n_of_images = 1
    elif file_extension == '.lif':        
        lif = LifFile(file_path)        
        img_list = [i for i in lif.get_iter_image()]    
        n_of_images = len(img_list)
    elif file_extension == '.tif':  
        n_of_images = 1
    return n_of_images

def read_tiff_channels(file_path, num_channels, num_layers):
    # Открываем TIFF файл
    tiff_image = Image.open(file_path)

    # Проверяем количество кадров (изображений) в файле
    n_frames = tiff_image.n_frames
    if n_frames != num_channels * num_layers:
        raise ValueError(f"Количество кадров в файле ({n_frames}) не соответствует заданным каналам ({num_channels}) и слоям ({num_layers})")

    # Инициализируем списки для каждого канала
    channels = [[] for _ in range(num_channels)]

    # Считываем все изображения и распределяем по каналам
    for i in range(n_frames):
        tiff_image.seek(i)
        frame = np.array(tiff_image.copy())
        channel_index = i % num_channels
        channels[channel_index].append(frame)

    # Преобразуем списки в numpy массивы для удобства работы
    channels = [np.array(channel) for channel in channels]

    # Закрываем файл
    tiff_image.close()

    return channels

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

def extract_image_stock(file_path, location, slice_start, slice_end, target_ch, dapi_ch):    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))    
    file_extension = splitext(file_path)[1].lower()    
    if file_extension == '.czi':
        # for the accumulation standard
        combined_image_s = []
        im_index = 0
        
        with CziFile(file_path) as czi:
            image_data = czi.asarray()                                
            if image_data.shape[2] == 3:
                dapi_ch = 2            
            slide = list(range(slice_start-1, slice_end))            
            sample_slice_1 = np_max(image_data[0, 0, target_ch, slide, :, :, 0], axis=0)
            sample_slice_3 = np_max(image_data[0, 0, dapi_ch, slide, :, :, 0], axis=0)
            
            synaptotag_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_synaptotag.png")
            save_image(sample_slice_1, synaptotag_file_path)    
            combined_image = zeros((*sample_slice_1.shape, 3), dtype='uint8')
            sample_slice_1_normalized = (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1)) * 255
            sample_slice_3_normalized = (sample_slice_3 - np.min(sample_slice_3)) / (np.max(sample_slice_3) - np.min(sample_slice_3)) * 255        
            combined_image[:, :, 0] = sample_slice_1_normalized  # RED synaptotagmin channel
            combined_image[:, :, 2] = sample_slice_3_normalized  # BLUE cell-label DAPI channel
            
            combined_image_s.append(combined_image)
        
    elif file_extension == '.lif':        
        lif = LifFile(file_path)        
        img_list = [i for i in lif.get_iter_image()]
        
        combined_image_s = []
        for im_index, image in enumerate(img_list):    
            slide = list(range(slice_start-1, slice_end))    
            
            frames_1 = collect_all_frames(image, target_ch)
            frames_2 = collect_all_frames(image, dapi_ch)
                        
            sample_slice_1 = np.max(frames_1, axis=0)
            sample_slice_3 = np.max(frames_2, axis=0)  
            
            # lif_name = lif.image_list[im_index]['name']
            synaptotag_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_synaptotag.png")
            save_image(sample_slice_1, synaptotag_file_path)    
            combined_image = zeros((*sample_slice_1.shape, 3), dtype='uint8')
            sample_slice_1_normalized = (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1)) * 255
            sample_slice_3_normalized = (sample_slice_3 - np.min(sample_slice_3)) / (np.max(sample_slice_3) - np.min(sample_slice_3)) * 255        
            combined_image[:, :, 0] = sample_slice_1_normalized  # synaptotagmin channel
            combined_image[:, :, 2] = sample_slice_3_normalized  # cell-label dapi channel
    
            combined_image_s.append(combined_image)
    elif file_extension == '.tif':
        # for the accumulation standard
        combined_image_s = []
        im_index = 0
        
        # Number of channels and depths
        num_channels = 4
        num_layers = 11
        
        # Read channels from a TIFF file
        channels = read_tiff_channels(file_path, num_channels, num_layers)
        # Get arrays frames_1 and frames_2
        frames_1 = channels[target_ch]
        frames_2 = channels[dapi_ch]
        
        # z-stack
        sample_slice_1 = np.max(frames_1, axis=0)
        sample_slice_3 = np.max(frames_2, axis=0)
        
        synaptotag_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_synaptotag.png")
        save_image(sample_slice_1, synaptotag_file_path)  
        
        # Normalizing images
        sample_slice_1_normalized = (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1)) * 255
        sample_slice_3_normalized = (sample_slice_3 - np.min(sample_slice_3)) / (np.max(sample_slice_3) - np.min(sample_slice_3)) * 255
        
        # Creating a composite image
        combined_image = np.zeros((*sample_slice_1.shape, 3), dtype='uint8')
        combined_image[:, :, 0] = sample_slice_1_normalized.astype('uint8')  # synaptotagmin channel
        combined_image[:, :, 2] = sample_slice_3_normalized.astype('uint8')  # cell-label dapi channel
        
        combined_image_s.append(combined_image)
        
    return combined_image_s

def process_file(file_path, location, slice_start, slice_end, target_ch, dapi_ch):
    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    
    combined_image_s = extract_image_stock(file_path, location, slice_start, slice_end, target_ch, dapi_ch)
    
    roi_coords_path_s = []
    image_file_path_s = []      
    
    n_of_images = filetype_checking(file_path)
    print(n_of_images)
    for im_index, combined_image in enumerate(combined_image_s):
        height, width = combined_image.shape[:2]
        dpi = 200
        figsize = width / float(dpi), height / float(dpi)
        
        fig, ax = subplots(figsize=(5, 5), dpi=dpi * 0.8)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        ax.imshow(combined_image)
        
        ax.text(0.5, 0.95, f"{file_path} - {location}", transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))
        ax.axis('off')
    
        coords = []
    
        def onselect(verts):
            coords.extend(verts)
            polygon = Polygon(verts, closed=True, edgecolor='#1DE720', facecolor='none', linewidth=2, alpha=0.7)
            ax.add_patch(polygon)
            draw()
            close(fig)
        
        props = dict(color='#1DE720', linestyle='-', linewidth=2, alpha=0.7)
        polygon_selector = PolygonSelector(ax, onselect, props=props)
        show(block=True)
    
    
        roi_coords_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_roi_coords.csv")
        coords_df = DataFrame(coords, columns=['x', 'y'])
        coords_df.to_csv(roi_coords_path, sep=';', index=False)
        # print(f"Coordinates saved to {roi_coords_path}")
    
        height, width = combined_image.shape[:2]
        dpi = 200
        figsize = width / float(dpi), height / float(dpi)
        
        fig, ax = subplots(figsize=figsize, dpi=dpi)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы
        ax.imshow(combined_image)
        polygon = Polygon(coords, closed=True, edgecolor='#1DE720', facecolor='none', linewidth=2, alpha=0.7)
        ax.add_patch(polygon)
        ax.text(0.5, 0.95, f"{file_path} - {location}", transform=ax.transAxes, fontsize=5, color='white', ha='center', va='top', bbox=dict(facecolor='black', alpha=0.5))
        ax.axis('off')
        image_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_with_roi.png")
        savefig(image_file_path, bbox_inches='tight', pad_inches=0)
        close(fig)
        
        roi_coords_path_s.append(roi_coords_path)
        image_file_path_s.append(image_file_path)
        
    # Объединение списков в один
    combined_file_list = roi_coords_path_s.copy()  # Создаем копию первого списка
    combined_file_list.extend(image_file_path_s)   # Добавляем элементы второго списка
    
    return combined_file_list
    
def filter_after_roi_selection(filter_radius, file_path, location):
    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
       
    denoised_image_path_s = []
    n_of_images = filetype_checking(file_path)
    for im_index in range(n_of_images):
        # print(im_index)
        synaptotag_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_synaptotag.png")
        sample_slice_1 = read_image(synaptotag_file_path)
            
        # Convert image to uint8 format
        if sample_slice_1.dtype != np.uint8:
            sample_slice_1 = (255 * (sample_slice_1 - np.min(sample_slice_1)) / (np.max(sample_slice_1) - np.min(sample_slice_1))).astype(np.uint8)
        
        # Let's make sure the image is two-dimensional
        if len(sample_slice_1.shape) > 2:
            sample_slice_1 = cv2.cvtColor(sample_slice_1, cv2.COLOR_BGR2GRAY)
            
        # Remove background
        background = cv2.medianBlur(sample_slice_1, filter_radius)
        denoised_image = cv2.subtract(sample_slice_1, background)
                
        denoised_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_denoised.png")
        save_image(denoised_image, denoised_image_path)
        
        denoised_image_path_s.append(denoised_image_path)       
        
    return denoised_image_path_s

def save_image(image, path):
    cv2.imwrite(path, image)
  
def read_image(path):
    return cv2.imread(path)

def max_entropy_threshold(image):
    hist, bin_edges = histogram(image.ravel(), bins=256, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    entropy = -hist * log(hist + finfo(float).eps)
    threshold = bin_mids[argmax(entropy)]
    return threshold

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
    
def binarize_images(file_path, row, binarization_method='max_entropy', min_size=64, max_size=100, pixel_to_micron_ratio = 0.12):    
    image_path = row['filepath']
    base_name = splitext(basename(image_path))[0]
    experiment_date = basename(dirname(image_path))
    
    masks_image_path_s = []
    n_of_images = filetype_checking(file_path)
    for im_index in range(n_of_images):
        
        denoised_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_denoised.png")
        masks_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_masks_roi_crop.png")
        full_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_full_roi_result_table.xlsx")
        summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_summary_roi_result_table.xlsx")
        roi_coords_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_roi_coords.csv")
        roi_mask_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_roi_mask.png")
    
        image = Image.open(denoised_image_path).convert('L')
        image_array = array(image)
    
        # Check if ROI coordinates file exists
        if exists(roi_coords_path):
            roi_coords_df = read_csv(roi_coords_path, delimiter=';')
            roi_coords = roi_coords_df[['x', 'y']].values
            roi_path = Path(roi_coords)
        
            x, y = np.meshgrid(np.arange(image_array.shape[1]), np.arange(image_array.shape[0]))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            roi_mask = roi_path.contains_points(points).reshape(image_array.shape)
        else:
            print("Warning: ROI coordinates file not found. Creating a mask the size of the entire image.")
            roi_mask = np.ones(image_array.shape, dtype=bool)
        
        # Save ROI mask as PNG image
        roi_mask_pil = Image.fromarray((roi_mask * 255).astype('uint8'))
        roi_mask_pil.save(roi_mask_image_path)
        
        threshold_value = get_threshold_value(image_array, binarization_method)
    
        binary_image = image_array > threshold_value
        binary_image = remove_small_objects(binary_image, min_size=min_size)
        binary_image = remove_large_objects(binary_image, max_size=max_size)
        
        binary_image_roi = binary_image & roi_mask
    
        binary_image_pil = Image.fromarray((binary_image_roi * 255).astype('uint8'))
        binary_image_pil = ImageOps.invert(binary_image_pil)
        binary_image_pil.save(masks_image_path)
        
        # distances = extract_scaling_distances_from_czi(file_path)
        # pixel_to_micron_ratio = distances['X']*1_000_000
        
        process_properties(image_array, binary_image_roi, roi_mask, pixel_to_micron_ratio, binarization_method, masks_image_path, full_result_path, summary_result_path)
        
        masks_image_path_s.append(masks_image_path)
        
    return masks_image_path_s

def remove_large_objects(ar, max_size):
    # Label connected components
    labeled = label(ar)
    for region in regionprops(labeled):
        if region.area > max_size:
            ar[labeled == region.label] = 0
    return ar

def process_properties(image_array, binary_image_roi, roi_mask, pixel_to_micron_ratio, binarization_method, masks_image_path, full_result_path, summary_result_path):

    labeled_image = label(binary_image_roi)
    props = regionprops(labeled_image, intensity_image=image_array)
                
    max_size = 500
    
    results = []
    total_objects = 0
    total_area = 0
    total_mean_intensity = 0
    roi_area = roi_mask.sum() * pixel_to_micron_ratio**2

    for index, prop in enumerate(props, start=1):
        area_microns = prop.area * pixel_to_micron_ratio**2
        if area_microns > max_size:
           continue  # Skip objects larger than max_size
        total_objects += 1
        total_area += area_microns
        total_mean_intensity += prop.mean_intensity * area_microns
        results.append({
            "": index,
            "Area": f"{area_microns:.3f}",
            "Mean": f"{prop.mean_intensity:.3f}",
            "Min": int(prop.min_intensity),
            "Max": int(prop.max_intensity)
        })

    results_df = DataFrame(results)
    results_df.to_excel(full_result_path, index=False)

    if total_objects > 0:
        average_size = total_area / total_objects
        average_mean_intensity = total_mean_intensity / total_area
    else:
        average_size = 0
        average_mean_intensity = 0

    summary_result = {
        "": 1,
        "Slice": basename(masks_image_path),
        "Count": total_objects,
        "Total Area": f"{total_area:.3f}",
        "Average Size": f"{average_size:.3f}",
        "%Area": f"{(total_area / roi_area) * 100:.3f}",
        "Mean": f"{average_mean_intensity:.3f}",
        "Binarization method": binarization_method
    }

    summary_df = DataFrame([summary_result])
    summary_df.to_excel(summary_result_path, index=False)


def remove_ccp(df, csv_file_path, rows_to_process, dpi=200, pixel_to_micron_ratio = 0.12):
    
    def onselect(verts):
        coords.extend(verts)
        polygon = Polygon(verts, closed=True, edgecolor='#1DE720', facecolor='none', linewidth=2, alpha=0.7)
        ax.add_patch(polygon)
        draw()
        close(fig)

    error_files = []
    for row_number_to_process in rows_to_process:
        try:
            row = df.iloc[row_number_to_process]
            if row['take_to_stat'] == 'no':
                print(f"Skipping row {row_number_to_process} because take_to_stat is 'no'")
                continue
            
            image_path = row['filepath']
            base_name = splitext(basename(image_path))[0]
            experiment_date = basename(dirname(image_path))
            
            n_of_images = filetype_checking(image_path)
            for im_index in range(n_of_images):                
                denoised_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_denoised.png")
                masks_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_masks_roi_crop.png")
                full_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_full_roi_result_table.xlsx")
                summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_summary_roi_result_table.xlsx")
                roi_mask_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_roi_mask.png")
    
                image = Image.open(denoised_image_path).convert('L')
                image_array = array(image)
                
                previous_masked_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_masks_roi_crop.png")
                roi_mask_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_roi_mask.png")
                print('Removing bad spots in ' + roi_mask_image_path)
                
                # reading file into previous_masked_image variable from previous_masked_image_path
                previous_masked_image = Image.open(previous_masked_image_path).convert('L')
                previous_masked_array = np.array(previous_masked_image)
                previous_masked_binary = (previous_masked_array > 0).astype(np.uint8)  # перевод в бинарный вид
                
                # read mask and convert to binary form
                mask_image = Image.open(roi_mask_image_path).convert('L')
                mask_array = np.array(mask_image)
                previous_roi_mask = (mask_array > 0).astype(np.uint8)
                
                # displaying the image
                fig, ax = subplots(figsize=(5, 5), dpi=dpi * 0.8)
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # убираем отступы
                
                ax.imshow(previous_masked_image, cmap='gray')
                
                # draw a polygon
                coords = []
                props = dict(color='#1DE720', linestyle='-', linewidth=2, alpha=0.7)
                polygon_selector = PolygonSelector(ax, onselect, props=props)
                show(block=True)
                
                # create a new mask from polygon coordinates
                x, y = np.meshgrid(np.arange(previous_masked_array.shape[1]), np.arange(previous_masked_array.shape[0]))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                polygon = Polygon(coords)
                new_roi_mask = polygon.contains_points(points).reshape(previous_masked_array.shape).astype(np.uint8)
                
                # applying error correction to the image
                corrected_image = previous_masked_binary | new_roi_mask
                
                # applying error correction to the mask
                corrected_mask = previous_roi_mask & ~new_roi_mask
                
                # saving a new mask
                new_mask_image_pil = Image.fromarray((corrected_mask * 255).astype('uint8'))
                new_mask_image_pil.save(roi_mask_image_path)
                
                # saving the image passed through the mask
                new_masked_image_pil = Image.fromarray((corrected_image * 255).astype('uint8'))
                new_masked_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_{im_index}_masks_roi_crop.png")
                new_masked_image_pil.save(new_masked_image_path)
                
                summary_data = read_excel(summary_result_path)
                binarization_method = summary_data['Binarization method'].values[0]
                
                # distances = extract_scaling_distances_from_czi(image_path)
                # pixel_to_micron_ratio = distances['X']*1_000_000
                
                # resave properties
                process_properties(image_array, ~corrected_image, ~corrected_mask, pixel_to_micron_ratio, binarization_method, masks_image_path, full_result_path, summary_result_path)
    

        except Exception as e:
            print(f"Error processing row {row_number_to_process} for file {image_path}: {e}")
            error_files.append(image_path)
    
    if error_files:
        print("\nErrors occurred in the following files:")
        for error_file in error_files:
            print(error_file)
    else:
        print("\nNo errors occurred during processing.")
import os
# Function to combine images and save them
def combine_and_save_images(image1_path, image2_path, image3_path, output_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image3 = Image.open(image3_path)
    
    # print('test 1')
    # Determine the size of the new image
    combined_width = image1.width + image2.width + image3.width
    combined_height = max(image1.height, image2.height, image3.height)
    
    # Create the new image
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image3, (0, 0))  # Paste the ROI image first
    combined_image.paste(image1, (image3.width, 0))  # Then the denoised image
    combined_image.paste(image2, (image3.width + image1.width, 0))  # Then the masks image
    
    # print('test 2')
    print(output_path)
    makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_image.save(output_path)

def combine_images(file_path, output_directory):
    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    
    combined_image_path_s = []
    n_of_images = filetype_checking(file_path)
    for im_index in range(n_of_images):            
        # Paths to the images
        denoised_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_denoised.png")
        masks_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_masks_roi_crop.png")
        roi_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_with_roi.png")
    
        # Combine and save images
        combined_image_path = join(output_directory, f"{experiment_date}_{base_name}_{im_index}_combined.png")
        
        combine_and_save_images(denoised_image_path, masks_image_path, roi_image_path, combined_image_path)
        
        combined_image_path_s.append(combined_image_path)
        
    return combined_image_path_s
    
    
def pp_one(file_path, row, output_directory):
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    
    summary_result_path_s = []
    summary_data_s = []
    n_of_images = filetype_checking(file_path)
    for im_index in range(n_of_images):   
        # Path to the results file
        summary_result_path = join(dirname(file_path), f"{experiment_date}_{base_name}_{im_index}_summary_roi_result_table.xlsx")
        
        # Check if the file exists
        if not exists(summary_result_path):
            print('test')
            print(f"File not found: {summary_result_path}")
            return None
        
        # Read the results file
        summary_data = read_excel(summary_result_path)
    
        if summary_data is not None:
            summary_data['filepath'] = file_path
            summary_data['im_index'] = im_index
            summary_data['location'] = row['location']
            summary_data['Postnatal_Age'] = row['Postnatal_Age']
            summary_data['Experiment_Number'] = row['Experiment_Number']
            
        summary_result_path_s.append(summary_result_path)
        summary_data_s.append(summary_data)
        
    return summary_data_s, summary_result_path_s
    
