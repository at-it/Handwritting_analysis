import cv2
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np

lastx, lasty = None, None
image_number = 0


def create_main_window() -> Tk():
    root = Tk()
    root.resizable(0, 0)
    root.title('Handwritten Digit Recongition GUI')
    return root


def create_canvas(root) -> Canvas:
    cv = Canvas(root, width=480, height=320, bg='white')
    cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)
    return cv


def clear_canvas(cv: Canvas) -> None:
    cv.delete('all')


def draw_lines(event, cv: Canvas) -> None:
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black',
                   capstyle=ROUND, smooth=TRUE, splinesteps=12)
    lastx, lasty = x, y


def activate_event(event, cv: Canvas) -> None:
    global lastx, lasty
    cv.bind('<B1-Motion>', lambda event: draw_lines(event, cv=cv))
    lastx, lasty = event.x, event.y


def bind_events(cv) -> None:
    cv.bind('<Button-1>', lambda event: activate_event(event, cv=cv))


def recognize_digit(root: Tk, cv: Canvas, model) -> None:
    global image_number

    filename = f'src/images/image_{image_number}.png'

    # get the widget coordinates
    x = root.winfo_rootx() + cv.winfo_x()
    y = root.winfo_rooty() + cv.winfo_y()

    x1 = x + cv.winfo_width()
    y1 = y + cv.winfo_height()

    # grab the image, crop it according to my requirement and save it in png
    ImageGrab.grab().crop((x, y, x1, y1)).save(filename)

    # read the image in color format
    image = cv2.imread(filename, cv2.IMREAD_COLOR)

    # converting to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply Otsu thresholding
    ret, th = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # find contour
    contours = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        # get bounding box and extract ROI
        x, y, w, h = cv2.boundingRect(cnt)
        # create rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 1)
        top = int(0.05 * th.shape[0])
        bottom = top
        left = int(0.05 * th.shape[1])
        right = left
        th_up = cv2.copyMakeBorder(
            th, top, bottom, left, right, cv2.BORDER_REPLICATE)
        # extract the image ROI
        roi = th[y-top:y+h+bottom, x-left:x+w+right]
        # resize roi image to 28 x 28 pixels
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # reshaping the image to support our model input
        img = img.reshape(1, 28, 28, 1)
        # normalizing the image to support our model input
        img = img/255.0
        # predicting the result
        pred = model.predict([img])[0]
        final_pred = np.argmax(pred)
        data = str(final_pred) + ' ' + str(int(max(pred)*100)) + '%'
        # cv2.putText() method is used to draw a text string on image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontscale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y-5), font, fontscale, color, thickness)

    # Showing the predicted results on new window
    cv2.imshow('image', image)
    cv2.waitKey(0)


def add_buttons_labels(root, cv, model) -> None:
    btn_save = Button(root, text='Recognize Digit',
                      command=lambda: recognize_digit(root, cv, model))
    btn_save.grid(row=2, column=0, pady=1, padx=1)
    btn_clear = Button(root, text='Clear window', command=lambda: clear_canvas(cv))
    btn_clear.grid(row=2, column=1, pady=1, padx=1)
