from os.path import splitext, basename, dirname, join
from os import makedirs
from pandas import read_csv, DataFrame, concat
from numpy import zeros, min as np_min, max as np_max, array, arange, meshgrid, vstack, histogram, finfo, log, argmax, asarray, mean
from matplotlib.pyplot import subplots, show, close, savefig, draw
from matplotlib.widgets import PolygonSelector
from matplotlib.patches import Polygon
from czifile import CziFile
from PIL import Image, ImageOps
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_isodata, threshold_mean, threshold_minimum
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from matplotlib.path import Path
from tqdm import tqdm


def process_file(file_path, location, slice_start, slice_end):
    with CziFile(file_path) as czi:
        image_data = czi.asarray()
        print("Image shape:", image_data.shape)
        channel_1 = 0 # synaptotagmin channel
        channel_3 = 3 # cell-label channel

        if image_data.shape[2] == 3:
            channel_3 = 2
            print("3 channels instead of 4")
        
        # slice_start = 2
        # slice_end = 6
        
        slide = list(range(slice_start, slice_end+1))
        
        sample_slice_1 = mean(image_data[0, 0, channel_1, slide, :, :, 0], axis = 0)        
        sample_slice_3 = mean(image_data[0, 0, channel_3, slide, :, :, 0], axis = 0)

        combined_image = zeros((*sample_slice_1.shape, 3), dtype='uint8')
        sample_slice_1_normalized = (sample_slice_1 - np_min(sample_slice_1)) / (np_max(sample_slice_1) - np_min(sample_slice_1)) * 255
        sample_slice_3_normalized = (sample_slice_3 - np_min(sample_slice_3)) / (np_max(sample_slice_3) - np_min(sample_slice_3)) * 255

        combined_image[:, :, 0] = sample_slice_1_normalized # synaptotagmin channel
        combined_image[:, :, 2] = sample_slice_3_normalized # cell-label channel

    height, width = combined_image.shape[:2]
    dpi = 200
    figsize = width / float(dpi), height / float(dpi)
    
    fig, ax = subplots(figsize=(5, 5), dpi=dpi*0.8)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Убираем отступы
    
    ax.imshow(combined_image)
    
    # Добавление текста на изображение
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

    base_name = splitext(basename(file_path))[0]
    experiment_date = basename(dirname(file_path))
    coords_file_path = join(dirname(file_path), f"{experiment_date}_{base_name}_roi_coords.csv")

    coords_df = DataFrame(coords, columns=['x', 'y'])
    coords_df.to_csv(coords_file_path, sep=';', index=False)
    print(f"Coordinates saved to {coords_file_path}")

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
    print(f"Image with ROI saved to {image_file_path}")

def max_entropy_threshold(image):
    hist, bin_edges = histogram(image.ravel(), bins=256, density=True)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]
    
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    entropy = -hist * log(hist + finfo(float).eps)
    threshold = bin_mids[argmax(entropy)]
    return threshold

def binarize_images(df, csv_file_path, rows_to_process, binarization_method='max_entropy', min_size=64, pixel_to_micron_ratio = 0.1):
    # pixel_to_micron_ratio Коэффициент перевода из пикселей в микрометры
    # df = read_csv(csv_file_path, delimiter=';')
    
    # rows_to_process = [row - 2 for row in rows_to_process]# формирование из эксель в нормальный стандарт
    
    error_files = []

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

                denoised_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
                masks_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.tif")
                full_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_full_roi_result_table.xls")
                summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xls")
                roi_coords_path = join(dirname(image_path), f"{experiment_date}_{base_name}_roi_coords.csv")

                image = Image.open(denoised_image_path).convert('L')
                image_array = array(image)

                roi_coords_df = read_csv(roi_coords_path, delimiter=';')
                roi_coords = roi_coords_df[['x', 'y']].values
                roi_path = Path(roi_coords)

                x, y = meshgrid(arange(image_array.shape[1]), arange(image_array.shape[0]))
                x, y = x.flatten(), y.flatten()
                points = vstack((x, y)).T
                roi_mask = roi_path.contains_points(points).reshape(image_array.shape)
                
                threshold_value = get_threshold_value(image_array, binarization_method)

                binary_image = image_array > threshold_value
                binary_image = remove_small_objects(binary_image, min_size=min_size)

                binary_image_roi = binary_image & roi_mask

                binary_image_pil = Image.fromarray((binary_image_roi * 255).astype('uint8'))
                binary_image_pil = ImageOps.invert(binary_image_pil)
                binary_image_pil.save(masks_image_path)

                labeled_image = label(binary_image_roi)
                props = regionprops(labeled_image, intensity_image=image_array)

                results = []
                total_objects = 0
                total_area = 0
                total_mean_intensity = 0
                roi_area = roi_mask.sum() * pixel_to_micron_ratio**2

                for index, prop in enumerate(props, start=1):
                    area_microns = prop.area * pixel_to_micron_ratio**2
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
                results_df.to_csv(full_result_path, sep='\t', index=False)

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
                    "Mean": f"{average_mean_intensity:.3f}"
                }

                summary_df = DataFrame([summary_result])
                summary_df.to_csv(summary_result_path, sep='\t', index=False)

                print(f"Row {row_number_to_process} processed. Saved binary image as {masks_image_path}.")
                pbar.update(1)

            except Exception as e:
                print(f"Error processing row {row_number_to_process} for file {image_path}: {e}")
                error_files.append(image_path)
                pbar.update(1)

    if error_files:
        print("\nErrors occurred in the following files:")
        for error_file in error_files:
            print(error_file)
    else:
        print("\nNo errors occurred during processing.")

# Function to combine images and save them
def combine_and_save_images(image1_path, image2_path, image3_path, output_path):
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)
    image3 = Image.open(image3_path)
    
    # Determine the size of the new image
    combined_width = image1.width + image2.width + image3.width
    combined_height = max(image1.height, image2.height, image3.height)
    
    # Create the new image
    combined_image = Image.new('RGB', (combined_width, combined_height))
    combined_image.paste(image3, (0, 0))  # Paste the ROI image first
    combined_image.paste(image1, (image3.width, 0))  # Then the denoised image
    combined_image.paste(image2, (image3.width + image1.width, 0))  # Then the masks image
    
    # Save the combined image
    combined_image.save(output_path)

def postprocess(df, csv_file_path, output_directory, rows_to_process):
    
    # Создаем выходную директорию, если ее нет
    makedirs(output_directory, exist_ok=True)

    # Загружаем данные из CSV
    # df = pd.read_csv(csv_file_path, delimiter=';')
    
    # Collect data and combine images
    summary_data_list = []
    
    # Добавляем прогресс-бар
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
                
                # Paths to the images
                denoised_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_denoised_Zprojection_crop.tif")
                masks_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_masks_roi_crop.tif")
                roi_image_path = join(dirname(image_path), f"{experiment_date}_{base_name}_with_roi.png")
            
                # Path to the results file
                summary_result_path = join(dirname(image_path), f"{experiment_date}_{base_name}_summary_roi_result_table.xls")
            
                # Combine and save images
                combined_image_path = join(output_directory, f"{experiment_date}_{base_name}_combined.tif")
                try:
                    combine_and_save_images(denoised_image_path, masks_image_path, roi_image_path, combined_image_path)
                except Exception as e:
                    print(f"Error combining images for {image_path}: {e}")
                    pbar.update(1)
                    continue
            
                # Save the first row of the results table
                summary_data = read_csv(summary_result_path, sep='\t')
                if summary_data is not None:
                    summary_data['filepath'] = image_path
                    summary_data['location'] = row['location']
                    summary_data['threshold_method'] = row['threshold_method']
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
    
    print("Post-processing completed. Updated data saved to:", summary_output_path)
    