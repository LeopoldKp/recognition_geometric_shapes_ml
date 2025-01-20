import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Загрузка модели
model = load_model('shape_classifier.h5')

# Директория с тестовыми изображениями
test_dir = 'data/check'

# Размер изображений, который использовался для обучения модели
img_size = (64, 64)

# Названия классов (в данном случае совпадают с названиями папок)
class_names = ['circle', 'square', 'triangle']

# Функция для предсказания класса изображения
def predict_image(model, img_path):
    try:
        # Загрузка изображения
        img = load_img(img_path, target_size=img_size)
        img_array = img_to_array(img) / 255.0  # Нормализация
        img_array = np.expand_dims(img_array, axis=0)  # Добавление измерения батча

        # Предсказание
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)  # Уверенность модели в предсказании
        return predicted_class, confidence
    except Exception as e:
        return f"Error processing {img_path}: {e}", 0

# Список для хранения данных о предсказаниях
results = []

# Обработка изображений из поддиректорий
if not os.path.exists(test_dir):
    print(f"Directory '{test_dir}' not found!")
else:
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):  # Проверка, что это папка
            print(f"Обрабатывается папка: {class_dir}")
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.jpg', '.png')):  # Проверка формата файла
                    img_path = os.path.join(class_path, filename)
                    predicted_class, confidence = predict_image(model, img_path)
                    print(f"Файл: {filename} -> Предсказанный класс: {predicted_class}, Уверенность: {confidence:.2f}")
                    results.append((filename, confidence))
                else:
                    print(f"Файл {filename} пропущен (не поддерживаемый формат).")
        else:
            print(f"{class_dir} пропущен (это не папка).")

# Сортировка результатов по уверенности
results.sort(key=lambda x: x[1])

# Построение графика
if results:
    filenames = [item[0] for item in results]
    confidences = [item[1] for item in results]

    plt.figure(figsize=(10, 6))
    plt.bar(filenames, confidences)
    plt.xlabel('Изображения')
    plt.ylabel('Уверенность')
    plt.title('Уверенность модели в предсказаниях')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
else:
    print("Нет данных для отображения графика.")
