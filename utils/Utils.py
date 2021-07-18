import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


def train_data():
    # Load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Preprocessing steps as originally MNIST is in the form of 1d array
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    # convert class vectors to binary class matrices
    num_classes = 10
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes=num_classes)

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")

    # normalization step
    x_train /= 255
    x_test /= 255

    print("x_train shape: ", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    batch_size = 32
    epochs = 10

    model = define_sequential_model(num_classes, input_shape)

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy, optimizer="adam",
                  metrics=["accuracy"])

    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(x_test, y_test))
    print("The model has been successfully trained")

    model.save("datasets\mnist.h5")
    print("Saving the model as mnist.h5")

    evaluate_sequential_model(model, x_test, y_test)


def define_sequential_model(num_classes: int, input_shape: tuple):
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation=tf.nn.softmax))
    return model


def evaluate_sequential_model(model: Sequential, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: ", score[0])
    print("Test accuracy: ", score[1])


train_data()
