class Dataset:
    def __init__(self, X_train, X_test, y_train, y_test, X_val=None, y_val=None) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
