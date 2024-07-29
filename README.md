# Synapto Catcher User Guide

Welcome to the Synapto Catcher program! This guide will help you navigate through the features and functionalities of the software.

## Overview

Synapto Catcher is designed to process and analyze images. The program interface is divided into several sections, each dedicated to a specific step in the image processing workflow. The console at the bottom displays the progress and results of each processing stage, with links to the output files.

## Instructions

### 1. Select File

- **Select File (protocol or .csz):** Click the browse button and navigate to the directory containing your image files. Select the desired protocol or .csz file.

#### Working with protocol
The protocol allows to process a large number of images at once.
Currently, the protocol must necessarily contain the columns: **filepath, comment, location, Experiment_Number, take_to_stat, Postnatal_Age**. 
- **filepath:** The 'filepath' column must contain direct paths to the CZI file. For example “E:\iMAGES\Sp P11.5 3\Experiment-500.czi”.
- **Experiment_Number:** The 'Experiment_Number' column should contain the number of the experiment, e.g. 500.
- **take_to_stat:** The 'take_to_stat' column can be empty, but if there is the word 'no' in any line, the experiment on that line will be excluded from processing.

Columns **comment**, **location**, **Postnatal_Age** can be left empty.
### 2. Set Experiment Parameters

- **Experiment Number:** Enter the experiment number associated with the images.
- **slice start:** Specify the starting slice number.
- **slice end:** Specify the ending slice number.

### 3. Select ROI

- **1. Select ROI:** Click this button to select the Region of Interest (ROI) for your images.

### 4. Filter Images

- **Filter radius:** Enter the desired filter radius value.
- **2. Filter:** Click this button to apply the filter to the images based on the specified radius.

### 5. Binarize Images

- **Binarization Method:** Choose the binarization method (e.g., otsu) from the dropdown menu.
- **Min size of an object:** Specify the minimum size of objects to be considered.
- **Max size of an object:** Specify the maximum size of objects to be considered.
- **3. Binarize:** Click this button to binarize the images based on the chosen parameters.
- **Remove bad spots:** Click this button to remove unwanted spots from the binarized images.

### 6. Combine Images

- **Output Directory:** Specify the output directory where the processed images will be saved.
- **4. Combine images:** Click this button to combine the images as per the defined parameters.

### 7. Postprocess

- **5. Postprocess (Result table):** Click this button to generate the result table from the processed images.

## Console

The console at the bottom of the interface provides real-time updates on the progress of each processing stage. It displays messages indicating the successful completion of each step and provides links to the resulting files. You can click these links to open and view the results.

---

Follow these steps to efficiently process and analyze your images using Synapto Catcher. If you encounter any issues or have questions, please refer to the troubleshooting section or contact support.

---

### Example Workflow

1. **Select File:**
   - Click the browse button and select `E:/IMAGES/PS.1_slide7`.
   
2. **Set Experiment Parameters:**
   - Experiment Number: (Leave blank if not applicable)
   - slice start: 2
   - slice end: 6

3. **Select ROI:**
   - Click `1. Select ROI` and define your region of interest.

4. **Filter Images:**
   - Filter radius: 17
   - Click `2. Filter`.

5. **Binarize Images:**
   - Binarization Method: otsu
   - Min size of an object: 20
   - Max size of an object: 200
   - Click `3. Binarize`.
   - Click `Remove bad spots` if necessary.

6. **Combine Images:**
   - Output Directory: `E:/IMAGES/PS.1_slide7`
   - Click `4. Combine images`.

7. **Postprocess:**
   - Click `5. Postprocess (Result table)`.

The console will display the progress and provide links to the output files for review.
