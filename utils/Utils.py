from classes.dataset import Dataset
from tensorflow.keras.datasets import mnist


@staticmethod
def load_mnist_dataset() -> Dataset:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_val = None
    y_val = None
    return Dataset(x_train, x_test, y_train, y_test, x_val, y_val)
