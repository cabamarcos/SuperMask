import os
import shutil
import random
from Scripts.translate import translate

# Rutas
base_path = "./animals10"
train_dir = os.path.join(base_path, "train")
test_dir = os.path.join(base_path, "test")

# 🧹 Borrar si existen (recrea el split)
if os.path.exists(train_dir):
    shutil.rmtree(train_dir)
if os.path.exists(test_dir):
    shutil.rmtree(test_dir)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

random.seed(42)

# Recorrer las carpetas de clases
for class_name_it in os.listdir(base_path):
    full_class_path = os.path.join(base_path, class_name_it)

    if not os.path.isdir(full_class_path):
        continue
    if class_name_it in ["train", "test", "__pycache__"]:
        continue

    class_name_en = translate.get(class_name_it, class_name_it)

    images = [f for f in os.listdir(full_class_path) if os.path.isfile(os.path.join(full_class_path, f))]
    if len(images) == 0:
        print(f"⚠️  No hay imágenes en {class_name_it}")
        continue

    random.shuffle(images)
    split = int(0.8 * len(images))
    train_images = images[:split]
    test_images = images[split:]

    train_class_dir = os.path.join(train_dir, class_name_en)
    test_class_dir = os.path.join(test_dir, class_name_en)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(full_class_path, img), os.path.join(train_class_dir, img))
    for img in test_images:
        shutil.copy(os.path.join(full_class_path, img), os.path.join(test_class_dir, img))

    print(f"✅ Clase '{class_name_it}' → '{class_name_en}': {len(train_images)} train, {len(test_images)} test")

print("\n🎉 Dataset preparado correctamente y sobrescrito si existía.")
