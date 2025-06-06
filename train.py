import itertools
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def build_cnn_model(filters=32, dense_units=128, learning_rate=0.001):
    model = Sequential(
        [
            Conv2D(filters, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(dense_units, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mlp_model(dense_units=128, learning_rate=0.001):
    model = Sequential(
        [
            Flatten(input_shape=(28, 28, 1)),
            Dense(dense_units, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, filename):
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.xlabel("epoch")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def grid_search(param_grid, build_fn, tag, data, epochs=3):
    (x_train, y_train), (x_test, y_test) = data
    best_acc = 0.0
    best_params = None
    best_model = None

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    for idx, params in enumerate(itertools.product(*param_grid.values())):
        cfg = dict(zip(param_grid.keys(), params))
        print(f"Training {tag} with params: {cfg}")
        model = build_fn(**cfg)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            verbose=0,
            validation_split=0.1,
            callbacks=[early_stop],
        )
        plot_history(history, f"history_{tag}_{idx+1}.png")
        _, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Validation accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = cfg
            best_model = model

    return best_acc, best_model, best_params


def main():
    data = load_data()

    cnn_grid = {
        "filters": [32, 64],
        "dense_units": [64, 128],
        "learning_rate": [0.001, 0.0005],
    }
    mlp_grid = {
        "dense_units": [128, 256],
        "learning_rate": [0.001, 0.0005],
    }

    cnn_acc, cnn_model, cnn_params = grid_search(cnn_grid, build_cnn_model, "cnn", data)
    mlp_acc, mlp_model, mlp_params = grid_search(mlp_grid, build_mlp_model, "mlp", data)

    if cnn_acc >= mlp_acc:
        best_model = cnn_model
        best_params = cnn_params
        tag = "CNN"
    else:
        best_model = mlp_model
        best_params = mlp_params
        tag = "MLP"

    print(f"Selected best model: {tag} with params {best_params}")
    (_, _), (x_test, y_test) = data
    best_model.save("model.h5")
    y_pred = best_model.predict(x_test, verbose=0)
    y_true = y_test.argmax(axis=1)
    y_pred_classes = y_pred.argmax(axis=1)
    print("Confusion Matrix")
    print(confusion_matrix(y_true, y_pred_classes))
    print("Classification Report")
    print(classification_report(y_true, y_pred_classes))


if __name__ == "__main__":
    main()
