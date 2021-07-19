from utils import tools as t

#setting up basic model settings
num_classes = 10
input_shape = (28, 28, 1)

dataset = t.load_mnist_dataset()
t.preprocess_dataset(dataset)
t.define_sequential_model(num_classes, input_shape)
t.train_sequential(dataset, num_classes, input_shape)