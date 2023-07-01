import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib
from tkinter import ttk

from preprocess_data import *
from files_utils import *

matplotlib.use('TkAgg')

IMAGE_SIZE = (110, 110) # (width, height)
WINDOW_SIZE = (1250, 750) # (width, height)
MAIN_WINDOW_SIZE = (1400, 750) # (width, height)
IMAGE_CNT_BY_ROW = 10

COLUMN = ('Date', 'Clustering Setup', 'Image resize', 'Image_convert', 'Cell_size', 'Magnitude_threshold', 'n_bins', 'INDEX')

IMAGE_PATH = './data/image/bg-20k-train'
JSON_PATH = "./data/result"

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

    window = tk.Toplevel(root)
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

def make_row_tuple(filename, index):
    row = []
    filename = filename.split('.')[0]
    filename = filename.split(';')
    date = filename[0]
    cluster_setup = filename[1]
    hog_parameter = list(eval(filename[2]).values())
    row.append(date)
    row.append(cluster_setup)
    row += hog_parameter
    row.append(index)
    return tuple(row)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Main Window")
    
    canvas = tk.Canvas(root, width=MAIN_WINDOW_SIZE[0], height=MAIN_WINDOW_SIZE[1])
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(root, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    scrollable_frame = ttk.Frame(canvas)
    treeview_frame = ttk.Frame(scrollable_frame)
    treeview_frame.grid(row=0, column=0, sticky=tk.NSEW)
    
    treeview = ttk.Treeview(treeview_frame, columns=COLUMN, show='headings', height=100)

    s = ttk.Style()
    s.configure('Treeview', rowheight=30)

    button_frame = ttk.Frame(scrollable_frame, padding=40)
    button_frame.grid(row=0, column=1, sticky=tk.NSEW)

    for col in COLUMN:
        treeview.heading(col, text=col)
        treeview.column(col, width=150, anchor='center')
    treeview.pack(pady=10, padx=10)
    
    json_list = os.listdir(JSON_PATH)
    json_list.sort()

    print(len(json_list))
    for index, filename in enumerate(json_list):
        index = index + 1
        row = make_row_tuple(filename, index)
        treeview.insert('', tk.END, values=row, tags=('black' if index % 2 == 0 else 'gray'))
        button = tk.Button(button_frame, text=str(index), height=1,\
            command=lambda name=filename: create_image_viewer(IMAGE_PATH, get_dict_from_json(os.path.join(JSON_PATH, name))),\
            width=5)
        button.pack(padx=0, pady=1)
    
    treeview.tag_configure('black', background='black')
    treeview.tag_configure('gray', background='gray')
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(scrollregion=canvas.bbox("all"))
    
    root.mainloop()