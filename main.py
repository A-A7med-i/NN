import numpy as np
from model.dnn import DNN
from layer.layers import Dense
from optimizer.optimizer import GD
from loss.loss import CategoricalCrossEntropy
from sklearn.datasets import make_classification
from activations.activations import Relu, Softmax
from sklearn.model_selection import train_test_split

FEATURES = 10
CLASSES = 4
SAMPLES = 2000


def one_hot_encode(y, num_classes):
    """Converts a vector of labels to a one-hot encoded matrix."""
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot


def main():
    X, y = make_classification(
        n_samples=SAMPLES, n_features=FEATURES, n_classes=CLASSES, n_informative=3
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, shuffle=True, random_state=0
    )

    y_train_ohe = one_hot_encode(y_train, CLASSES)

    model = DNN()

    model.add(Dense(FEATURES, 16, Relu()))
    model.add(Dense(16, 32, Relu()))
    model.add(Dense(32, CLASSES, Softmax()))

    model.summary()

    model.compile(loss=CategoricalCrossEntropy(), optimizer=GD(learning_rate=0.1))

    history = model.train(X_train, y_train_ohe, epochs=100, batch_size=16)

    model.plot_history(history)


if __name__ == "__main__":
    main()
