from src.data_loader import load_data
from src.model import build_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
print("Loading data...")
X, y = load_data("data/train")

print("Data loaded:", X.shape)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build model
model = build_model()

# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)

# Save model
model.save("models/model.h5")
print("Model saved!")

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])
plt.savefig("outputs/accuracy_plot.png")

print("Training complete!")