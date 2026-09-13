import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)

print("\nLoading MNIST dataset...\n")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Training images:", X_train.shape)
print("Testing images:", X_test.shape)

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train,
    y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=64
)

loss, accuracy = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print("\nTest Accuracy:", accuracy)

predictions = model.predict(X_test)

predicted_labels = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predicted_labels
    )
)

cm = confusion_matrix(
    y_test,
    predicted_labels
)

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.title("MNIST Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")

plt.colorbar()

plt.xticks(range(10))
plt.yticks(range(10))

plt.tight_layout()

plt.savefig("graphs/confusion_matrix.png")

plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title("Training and Validation Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig("graphs/accuracy_graph.png")

plt.close()

model.save(
    "models/handwritten_character_model.keras"
)

print("\nModel saved successfully!")

print("\nHandwritten Character Recognition training completed!")
