from typing import Tuple
from tensorflow.keras.datasets import mnist
from classes.dataset import Dataset
from classes import dataset
import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


def load_mnist_dataset() -> Dataset:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_val = None
    y_val = None
    return Dataset(X_train, X_test, y_train, y_test, X_val, y_val)


def preprocess_dataset(dataset: Dataset):
    dataset.X_train = dataset.X_train.reshape(
        dataset.X_train.shape[0], 28, 28, 1)
    dataset.X_test = dataset.X_test.reshape(dataset.X_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # Adding class vectors to 0-9 digits
    num_classes = 10
    dataset.y_train = tensorflow.keras.utils.to_categorical(
        dataset.y_train, num_classes=num_classes)
    dataset.y_test = tensorflow.keras.utils.to_categorical(
        dataset.y_test, num_classes=num_classes)

    dataset.X_train = dataset.X_train.astype("float32")
    dataset.X_test = dataset.X_test.astype("float32")

    dataset.X_train /= 255
    dataset.X_test /= 255


def define_sequential_model(num_classes: int, input_shape: Tuple[int, int, int]):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train_sequential(dataset: dataset, num_classes: int, input_shape: Tuple[int, int, int]):

    # Set model settings
    batch_size = 100
    epochs = 20

    model = define_sequential_model(num_classes, input_shape)
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer="adam",
                  metrics=["accuracy"])
    hist = model.fit(dataset.X_train, dataset.y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                     validation_data=(dataset.X_test, dataset.y_test))
    model.save("src\model\mnist2.h5")

    evaluate_sequential_model(model, dataset.X_test, dataset.y_test)


def evaluate_sequential_model(model: Sequential, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])
