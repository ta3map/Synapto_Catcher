#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys, os
import easygui
from PIL import Image

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from graphical_processor import ParallelogramEditor, process_czi_image
from image_processor import transform_parallelogram_to_rectangle, plot_gray_histograms
    


# Opens a dialog box to select an image
image_path = easygui.fileopenbox(title="Select Image File", filetypes=["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.czi"])

if image_path is None:
    print("No file selected, exiting...")
    sys.exit(1)

import cv2

# If file is czi format
if image_path.endswith('.czi'):
    pil_img = process_czi_image(image_path, channels = [0,1,3])
    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
else:
    # Uploading an image
    image = cv2.imread(image_path)

# Initialize the class with an image
editor = ParallelogramEditor(image, scale_factor=0.8)
editor.run()

parallelogram_points = editor.get_coordinates()
if parallelogram_points:
    # Convert the parallelogram to a rectangle
    transformed_image_rectangle, rectangle_size, masks = transform_parallelogram_to_rectangle(image, parallelogram_points)
    # Constructing histograms and images on one figure
    plot_gray_histograms(transformed_image_rectangle, rectangle_size, bin_size=10, invert = True)

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys, os
import easygui
from PIL import Image
# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from graphical_processor import ParallelogramEditor

import pickle

# Загрузка данных из файла
with open('combined_image_1.pkl', 'rb') as file:
    combined_image = pickle.load(file)

image = combined_image[0]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

import matplotlib.pyplot as plt

# Разделение изображения на три канала
b_channel, g_channel, r_channel = cv2.split(image)

# Визуализация каждого канала отдельно
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(b_channel, cmap='gray')
plt.title('Blue channel')

plt.subplot(1, 3, 2)
plt.imshow(g_channel, cmap='gray')
plt.title('Green channel')

plt.subplot(1, 3, 3)
plt.imshow(r_channel, cmap='gray')
plt.title('Red channel')

plt.show()

plt.figure()
plt.imshow(image)
plt.show()

print("Shape of image:", image.shape)  # Должен быть (высота, ширина, 3)
print("Data type of image:", image.dtype)  # Обычно uint8
print("First pixel value:", image[0, 0])  # Вывод первого пикселя (значения должны быть в диапазоне 0-255)


editor = ParallelogramEditor(image, scale_factor=0.8)
editor.run()
