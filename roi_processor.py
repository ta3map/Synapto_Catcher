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



import pickle


def extract_image_stock(file_path, location, slice_start, slice_end):
    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    # print(f"{experiment_date}_{base_name}")
    
    with CziFile(file_path) as czi:
        image_data = czi.asarray()
        
        distances = extract_scaling_distances_from_czi(file_path)
        pixel_to_micron_ratio = distances['X']*1_000_000
        
        target_ch = 0  # synaptotagmin channel
        dapi_ch = 3  # cell-label channel
        
        # print("Image shape:", image_data.shape)
        # print('synaptotagmin channel:', target_ch)
        # print('cell-label channel:', dapi_ch)
        # print('slice start:', slice_start)
        # print('slice end:', slice_end)
        # print('pixel to micron ratio:', pixel_to_micron_ratio)
        
        if image_data.shape[2] == 3:
            dapi_ch = 2
            # print("3 channels instead of 4")
        
        slide = list(range(slice_start-1, slice_end))
        
        sample_slice_1 = np_max(image_data[0, 0, target_ch, slide, :, :, 0], axis=0)
        sample_slice_3 = np_max(image_data[0, 0, dapi_ch, slide, :, :, 0], axis=0)

        combined_image = zeros((*sample_slice_1.shape, 3), dtype='uint8')
        sample_slice_1_normalized = (sample_slice_1 - np_min(sample_slice_1)) / (np_max(sample_slice_1) - np_min(sample_slice_1)) * 255
        sample_slice_3_normalized = (sample_slice_3 - np_min(sample_slice_3)) / (np_max(sample_slice_3) - np_min(sample_slice_3)) * 255

        combined_image[:, :, 0] = sample_slice_1_normalized  # synaptotagmin channel
        combined_image[:, :, 2] = sample_slice_3_normalized  # cell-label channel
        
        synaptotag_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_synaptotag.png")
        save_image(sample_slice_1, synaptotag_file_path)
        
        return sample_slice_1, sample_slice_3, combined_image, synaptotag_file_path

def process_file(file_path, location, slice_start=2, slice_end=6):
    
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
        
    sample_slice_1, sample_slice_3, combined_image, _ = extract_image_stock(file_path, location, slice_start, slice_end)
        
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


    roi_coords_path = join(dirname(file_path), f"{experiment_date}_{base_name}_roi_coords.csv")
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
    image_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_with_roi.png")
    savefig(image_file_path, bbox_inches='tight', pad_inches=0)
    close(fig)
    # print(f"Image with ROI saved to {image_file_path}")
    
    return [roi_coords_path, image_file_path]


    
def filter_after_roi_selection(filter_radius, file_path, location):
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
       
    synaptotag_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_synaptotag.png")
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
            
    denoised_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_denoised.png")
    
    save_image(denoised_image, denoised_image_path)
    # print(f"Denoised image saved to {denoised_image_path}")
    return [denoised_image_path]

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
    
def binarize_images(file_path, row, binarization_method='max_entropy', min_size=64, max_size=100):    
    image_path = row['filepath']
    base_name = splitext(basename(image_path))[0]
    experiment_date = basename(dirname(image_path))

    denoised_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_denoised.png")
    masks_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.png")
    full_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_full_roi_result_table.xlsx")
    summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xlsx")
    roi_coords_path = join(dirname(image_path), f"{experiment_date}_{base_name}_roi_coords.csv")
    roi_mask_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_roi_mask.png")

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
    
    distances = extract_scaling_distances_from_czi(file_path)
    pixel_to_micron_ratio = distances['X']*1_000_000
    
    process_properties(image_array, binary_image_roi, roi_mask, pixel_to_micron_ratio, binarization_method, masks_image_path, full_result_path, summary_result_path)

    return [masks_image_path]

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


def remove_ccp(df, csv_file_path, rows_to_process, dpi=200):
    
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
            
            denoised_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_denoised.png")
            masks_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.png")
            full_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_full_roi_result_table.xlsx")
            summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xlsx")
            roi_mask_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_roi_mask.png")

            image = Image.open(denoised_image_path).convert('L')
            image_array = array(image)
            
            previous_masked_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.png")
            roi_mask_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_roi_mask.png")
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
            new_masked_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.png")
            new_masked_image_pil.save(new_masked_image_path)
            
            summary_data = read_excel(summary_result_path)
            binarization_method = summary_data['Binarization method'].values[0]
            
            distances = extract_scaling_distances_from_czi(image_path)
            pixel_to_micron_ratio = distances['X']*1_000_000
            
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
    
    # Paths to the images
    denoised_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_denoised.png")
    masks_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_masks_roi_crop.png")
    roi_image_path = join(dirname(file_path), f"{experiment_date}_{base_name}_with_roi.png")

    # Combine and save images
    combined_image_path = join(output_directory, f"{experiment_date}_{base_name}_combined.png")
    
    combine_and_save_images(denoised_image_path, masks_image_path, roi_image_path, combined_image_path)
    return [combined_image_path]
    
    
def pp_one(file_path, row, output_directory):
    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    # Path to the results file
    summary_result_path = join(dirname(file_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xlsx")
    
    # Check if the file exists
    if not exists(summary_result_path):
        print('test')
        print(f"File not found: {summary_result_path}")
        return None
    
    # Read the results file
    summary_data = read_excel(summary_result_path)
    
    if summary_data is not None:
        summary_data['filepath'] = file_path
        summary_data['location'] = row['location']
        summary_data['Postnatal_Age'] = row['Postnatal_Age']
        summary_data['Experiment_Number'] = row['Experiment_Number']
        
    return summary_data, summary_result_path
    
def postprocess(df, csv_file_path, output_directory, rows_to_process):
    
    # Create an output directory if there isn't one
    makedirs(output_directory, exist_ok=True)
    
    # Collect data and combine images
    summary_data_list = []
    
    # Add a progress bar
    with tqdm(total=len(rows_to_process), desc="Processing images") as pbar:
        for row_number_to_process in rows_to_process:
            try:
                row = df.iloc[row_number_to_process]
                if row['take_to_stat'] == 'no':
                    print(f"Skipping row {row_number_to_process} because take_to_stat is 'no'")
                    pbar.update(1)
                    continue
                
                image_path = row['filepath']
                base_name = splitext(basename(image_path))[0]
                experiment_date = basename(dirname(image_path))
                                
                # Path to the results file
                summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xlsx")
                # Save the first row of the results table
                summary_data = read_excel(summary_result_path)
                if summary_data is not None:
                    summary_data['filepath'] = image_path
                    summary_data['location'] = row['location']
                    # summary_data['threshold_method'] = row['threshold_method']
                    summary_data['Postnatal_Age'] = row['Postnatal_Age']
                    summary_data['Experiment_Number'] = row['Experiment_Number']
                    summary_data_list.append(summary_data)
                
                pbar.update(1)
            except Exception as e:
                print(f"Error processing row {row_number_to_process}: {e}")
                pbar.update(1)
    
    # Create a DataFrame with collected data
    summary_df = concat(summary_data_list, ignore_index=True)
    summary_df.drop(summary_df.columns[[0, -1]], axis=1, inplace=True)
    
    # summary_df['Postnatal_Age'] = asarray(df['Postnatal_Age'][rows_to_process])
    # summary_df['Experiment_Number'] = asarray(df['Experiment_Number'][rows_to_process])
        
    # Save the updated DataFrame to a new file
    summary_output_path = join(output_directory, "collected_roi_summary_data.xlsx")
    summary_df.to_excel(summary_output_path, index=False)
    
    print("Post-processing completed.", summary_output_path)
    return [summary_output_path]
    