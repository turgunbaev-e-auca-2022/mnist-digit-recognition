"""MNIST baseline and CNN, adapted from my academic notebook."""
import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model(kind):
    if kind == "baseline":
        return keras.Sequential([
            keras.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(10, activation="softmax"),
        ])
    return keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    keras.utils.set_random_seed(42)
    (images, labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
    images = images.astype("float32")[..., None] / 255.0
    test_images = test_images.astype("float32")[..., None] / 255.0
    # Keep 10,000 training examples for validation; never train on test data.
    train_images, val_images = images[:50000], images[50000:]
    train_labels, val_labels = labels[:50000], labels[50000:]

    output = Path("outputs")
    output.mkdir(exist_ok=True)
    results = {}
    for kind in ("baseline", "cnn"):
        keras.backend.clear_session()
        keras.utils.set_random_seed(42)
        model = build_model(kind)
        model.compile(optimizer="rmsprop",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        history = model.fit(train_images, train_labels,
                            validation_data=(val_images, val_labels),
                            epochs=args.epochs, batch_size=args.batch_size)
        loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
        model.save(output / f"{kind}.keras")
        results[kind] = {"test_loss": float(loss), "test_accuracy": float(accuracy),
                         "history": history.history}
        print(f"{kind}: test accuracy = {accuracy:.4%}")

    report = {"seed": 42, "epochs": args.epochs,
              "batch_size": args.batch_size, "tensorflow_version": tf.__version__,
              "train_size": 50000, "validation_size": 10000,
              "test_size": 10000, "models": results}
    (output / "metrics.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
