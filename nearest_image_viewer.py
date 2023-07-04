import os
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
import random

from preprocess_data import *
from files_utils import *
from hog import *

matplotlib.use('TkAgg')

IMAGE_PATH = './data/image/bg-20k-train'
DATA_PATH = './data/csv/' + get_config_string() + ';proportion' + '.csv'

def show_nearest_images(IMAGE_PATH, target_image_name, nearest_image_name_list):

    window = tk.Tk()
    window.title("Nearest Image Show")

    canvas = tk.Canvas(window, width=1200, height=700)
    canvas.pack()

    top_image = Image.open(os.path.join(IMAGE_PATH, target_image_name))
    top_image = top_image.resize((400, 300))
    top_image_tk = ImageTk.PhotoImage(top_image)

    canvas.create_image(550, 150, image=top_image_tk)


    image_frame = tk.Frame(canvas)
    canvas.create_window(0, 400, window=image_frame, anchor="nw")
    for i, image_data in enumerate(nearest_image_name_list):
        image = Image.open(os.path.join(IMAGE_PATH, image_data[1]))
        image = image.resize((210, 210))
        image_tk = ImageTk.PhotoImage(image)

        image_label = tk.Label(image_frame, image=image_tk)
        image_label.image = image_tk
        image_label.grid(row=i // 5, column=i % 5, padx=5, pady=5)

        text_label = tk.Label(image_frame, text=str(image_data[0]))
        text_label.grid(row=i // 5 + 1, column=i % 5, padx=5, pady=5)

    window.mainloop()

def nearest_image_viewer(IMAGE_PATH, DATA_PATH):
    name_list = get_name_data_list(DATA_PATH)
    random.shuffle(name_list)
    for target_image_name in name_list:
        nearest_image_name_list = get_nearest_images(target_image_name, get_raw_data_list_hog(DATA_PATH, False), get_name_data_list(DATA_PATH), 5)
        show_nearest_images(IMAGE_PATH, target_image_name, nearest_image_name_list)
    
if __name__ == '__main__':
    nearest_image_viewer(DATA_PATH, IMAGE_PATH)  