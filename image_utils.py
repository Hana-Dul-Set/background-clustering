import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib

from preprocess_data import *
from files_utils import *

matplotlib.use('TkAgg')

IMAGE_SIZE = (110, 110) # (width, height)
WINDOW_SIZE = (1250, 750) # (width, height)
IMAGE_CNT_BY_ROW = 10

IMAGE_PATH = './data/image/bg-20k-train'
JSON_PATH = "./data/result"

# deprecated
def show_10_images(image_filenames, IMAGE_PATH):

    fig, axes = plt.subplots(2, 5, figsize=(10, 5))

    for ax, filename in zip(axes.flatten(), image_filenames):
        image = plt.imread(os.path.join(IMAGE_PATH, filename))  
        ax.imshow(image)              
        ax.axis('off')

    plt.tight_layout()

    plt.show()

# deprectaed
def show_images_by_label(label_list, name_list, label_cnt):
    data_by_label_list = get_elements_list_by_label(label_list, name_list, label_cnt)

    for one_label_data_list in data_by_label_list:
        random_elements = get_random_elements(one_label_data_list, 10)
        show_10_images(random_elements)

def create_image_viewer(IMAGE_PATH, labeled_image_files):
    label_value = 0
    def load_labeled_images(label, labeled_image_files):
        nonlocal frame, label_value, label_value_label
        for widget in frame.winfo_children():
            widget.destroy()
        if label in labeled_image_files:
            image_files = labeled_image_files[label]
            image_count = 0
            for filename in image_files:
                image = Image.open(os.path.join(IMAGE_PATH, filename))
                image = image.resize(IMAGE_SIZE)
                photo = ImageTk.PhotoImage(image)
                label = tk.Label(frame, image=photo)
                label.image = photo
                label.grid(row=image_count // IMAGE_CNT_BY_ROW, column=image_count % IMAGE_CNT_BY_ROW, padx=5, pady=5)
                image_count += 1
        label_value_label.config(text=f"Label: {label_value}")

    def increment_label():
        nonlocal label_value
        label_value = (label_value + 1) % len(labeled_image_files)
        load_labeled_images(str(label_value), labeled_image_files)

    def decrement_label():
        nonlocal label_value
        label_value = (label_value - 1) % len(labeled_image_files)
        load_labeled_images(str(label_value), labeled_image_files)

    window = tk.Tk()
    window.title("Label Image Viewer")

    canvas = tk.Canvas(window, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
    canvas.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")

    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=frame, anchor="nw")

    increment_button = tk.Button(window, text="Up", command=increment_label)
    increment_button.pack(side="top")

    decrement_button = tk.Button(window, text="Down", command=decrement_label)
    decrement_button.pack(side="top")

    label_value_label = tk.Label(window, text=f"Label: {label_value}")
    label_value_label.pack() 
    
    load_labeled_images(str(label_value), labeled_image_files)

    window.mainloop()

if __name__ == '__main__':
    print(1)