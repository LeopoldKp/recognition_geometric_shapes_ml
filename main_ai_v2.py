import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from PIL import Image, ImageDraw

# Dataset class
class ShapeDataset(Sequence):
    def __init__(self, data_dir, batch_size, img_size):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.images = []
        self.labels = []

        # Class mapping for shapes
        self.shape_classes = {"square": 0, "circle": 1, "triangle": 2}

        for file_name in os.listdir(data_dir):
            if file_name.endswith('.png'):
                image_path = os.path.join(data_dir, file_name)
                label_path = os.path.join(data_dir, file_name.replace('.png', '.json'))

                if os.path.exists(label_path):
                    self.images.append(image_path)
                    with open(label_path, 'r') as f:
                        label_data = json.load(f)

                    # Convert labels to one-hot encoding
                    label_tensor = np.zeros(len(self.shape_classes))
                    for shape, _ in label_data.items():
                        if shape in self.shape_classes:
                            label_tensor[self.shape_classes[shape]] = 1.0
                    self.labels.append(label_tensor)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        for image_path in batch_images:
            image = Image.open(image_path).convert('RGB').resize(self.img_size)
            images.append(np.array(image) / 255.0)

        return np.array(images), np.array(batch_labels)

# Neural network
def create_model(img_size):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='sigmoid')  # 3 outputs for multi-label classification
    ])
    return model

# Annotating test images
def annotate_image(image_path, predictions, output_path):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    for shape, coords in predictions.items():
        if shape == 'square':
            draw.rectangle(coords, outline="red", width=2)
        elif shape == 'circle':
            draw.ellipse(coords, outline="blue", width=2)
        elif shape == 'triangle':
            draw.polygon(coords, outline="green", width=2)

    image.save(output_path)

# Paths and parameters
data_dir = 'dataset/train'
val_dir = 'dataset/validate'
test_dir = 'dataset/test'
img_size = (128, 128)
batch_size = 8

# Data generators
train_dataset = ShapeDataset(data_dir, batch_size, img_size)
val_dataset = ShapeDataset(val_dir, batch_size, img_size)

# Model, loss, and optimizer
model = create_model(img_size)
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=10)

# Annotate test images
test_dataset = ShapeDataset(test_dir, batch_size, img_size)
for idx in range(len(test_dataset)):
    images, labels = test_dataset[idx]
    predictions = model.predict(images)
    for i, prediction in enumerate(predictions):
        shape_predictions = {"square": labels[i][0], "circle": labels[i][1], "triangle": labels[i][2]}
        annotate_image(test_dataset.images[idx * batch_size + i], shape_predictions, f"output_{idx * batch_size + i}.png")
