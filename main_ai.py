from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Параметры
img_size = (64, 64)
num_classes = 3

# Создание модели
model = Sequential([
    Input(shape=(*img_size, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Компиляция модели
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Подготовка данных
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_generator = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
    'data/check',
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Чтобы порядок был предсказуем для оценки
)

# Ранняя остановка
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Обучение модели
history = model.fit(
    train_generator,
    epochs=1000,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# Оценка на тестовых данных
y_true = test_generator.classes  # Истинные классы
class_indices = {v: k for k, v in test_generator.class_indices.items()}  # Для отображения классов
y_pred_proba = model.predict(test_generator)
y_pred = np.argmax(y_pred_proba, axis=1)

# Расчёт метрик
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')

print(f"Test Metrics:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}")

# Матрица ошибок
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_indices.keys(), yticklabels=class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Визуализация обучения
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()

# Сохранение модели
model.save('shape_classifier.h5')





# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt

# # Параметры
# img_size = (64, 64)
# num_classes = 3

# # Создание модели
# model = Sequential([
#     Input(shape=(*img_size, 3)),
#     Conv2D(32, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax')
# ])

# # Компиляция модели с уменьшенной скоростью обучения
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# # Подготовка данных
# train_datagen = ImageDataGenerator(
#     rescale=1.0/255.0,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.2
# )

# train_generator = train_datagen.flow_from_directory(
#     'data/train',
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical',
#     subset='training'
# )

# val_generator = train_datagen.flow_from_directory(
#     'data/train',
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical',
#     subset='validation'
# )

# test_generator = ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(
#     'data/check',
#     target_size=(64, 64),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Ранняя остановка при отсутствии улучшений
# early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# # Обучение модели
# history = model.fit(
#     train_generator,
#     epochs=1000,
#     validation_data=val_generator,
#     callbacks=[early_stop]
# )

# # Оценка модели
# val_loss, val_acc = model.evaluate(val_generator)
# print(f"Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_acc:.2f}")
# test_loss, test_acc = model.evaluate(test_generator)
# print(f"Test Loss: {test_loss:.2f}, Test Accuracy: {test_acc:.2f}")

# # Визуализация обучения
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.title('Accuracy')
# plt.show()

# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.legend()
# plt.title('Loss')
# plt.show()

# # Сохранение модели
# model.save('shape_classifier.h5')
