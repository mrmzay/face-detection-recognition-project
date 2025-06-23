from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import torch
import os


facenet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=20)

def get_embedding(face_img):
    img_cropped = mtcnn(face_img)
    if img_cropped is None:
        return None
    embedding = facenet(img_cropped.unsqueeze(0))
    return embedding.detach().numpy()[0]


train_data = {
    "mohit_normal": ["mohit.jpg"],
    "mohit_masked": ["mohit_mask.jpg"],
    "mohit_capped": ["mohit_cap.jpg"]
}

X_train = []
y_train = []

for label, images in train_data.items():
    for img_path in images:
        img = Image.open(img_path)
        emb = get_embedding(img)
        if emb is not None:
            X_train.append(emb)
            y_train.append(label)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)


mohit_data = [
    ("mohit.jpg", "mohit_normal"),
    ("mohit.jpg", "mohit_masked"),
    ("mohit.jpg", "mohit_capped"),
]

correct = 0
total = 0

for img_path, actual in mohit_data:
    img = Image.open(img_path)
    emb = get_embedding(img)
    if emb is None:
        continue
    pred = knn.predict([emb])[0]
    print(f"Predicted: {pred}, Actual: {actual}")
    if pred == actual:
        correct += 1
    total += 1

accuracy = (correct / total) * 100 if total else 0
print(f"\nAccuracy: {accuracy:.2f}%")