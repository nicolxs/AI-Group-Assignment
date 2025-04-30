import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import random

# Constants
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  # 0–42

def load_data(data_dir):
    """
    Loads and preprocesses all images and labels from the GTSRB dataset.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        folder_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            try:
                image_path = os.path.join(folder_path, file)
                image = cv2.imread(image_path)

                if image is None:
                    continue

                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(category)
            except Exception as e:
                print(f"Error loading image: {file}, skipped. Error: {e}")
                continue

    return np.array(images), np.array(labels)


def get_model():
    """
    Defines and compiles the CNN model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_confusion_matrix(y_true, y_pred):
    """
    Draws and saves the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    plt.close()


def show_sample_predictions(model, X_test, y_test):
    """
    Displays sample test images and predictions.
    """
    print("Showing sample predictions...")

    indices = random.sample(range(len(X_test)), 5)
    class_names = list(range(NUM_CATEGORIES))  # just numbers 0-42

    for i in indices:
        img = X_test[i]
        true_label = np.argmax(y_test[i])
        prediction = model.predict(np.array([img]))
        predicted_label = np.argmax(prediction)

        plt.imshow(img)
        plt.title(f"True: {true_label}, Predicted: {predicted_label}")
        plt.axis('off')
        plt.show()


def main():
    # Check usage
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic_signs.py gtsrb_directory [model.h5]")

    data_dir = sys.argv[1]
    model_file = sys.argv[2] if len(sys.argv) == 3 else None

    # Load and prepare data
    print("Loading data...")
    images, labels = load_data(data_dir)

    print(f"Total samples: {len(images)}")
    images = images / 255.0  # Normalize to 0–1
    labels = to_categorical(labels, NUM_CATEGORIES)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Build and train model
    print("Training model...")
    model = get_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Model accuracy: {test_acc:.4f}")

    # Confusion matrix
    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred)

    # Show some predictions
    show_sample_predictions(model, X_test, y_test)

    # Save model
    if model_file:
        model.save(model_file)
        print(f"Model saved to {model_file}")


if __name__ == "__main__":
    main()
