import tensorflow as tf
from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
from pillow import ImageGrab, Image
import numpy as np
import win32gui

model = load_model("model/mnist.h5")

# creating main window

root = Tk()
root.resizable(0, 0)
root.title("Handwritten Digit Recongition GUI")

# initialize few variables

lastx, lasty = None, None
image_number = 0

# create canvas for drawing

cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

# Coding the events
cv.bind('<Button-1>', activate_event)

# Adding button and labels
btn_save = Button(text="Recognize Digit", command=recognize_digit)
btn_save.grid(row=2, column=0, pady=1, padx=1)
btn_clear = Button(text="Clear window", command=clear_window)
btn_clear.grid(row=2, column=1, pady=1, padx=1)

# Application start
root.mainloop()


def clear_window():
    global cv
    cv.delete("all")


def activate_event(event):
    global lastx, lasty
    cv.bind("<B1-Motion>", draw_lines)
    lastx, lasty = event.x, event.y


def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black',
                   capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y
