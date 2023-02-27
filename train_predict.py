import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
from PIL import Image, ImageDraw
from sklearn import neighbors
import os
import os.path
import math
import pickle

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def image_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        face_bounding_boxes = face_recognition.face_locations(image)
        if len(face_bounding_boxes) != 1:
            return None
        return face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
    except Exception as e:
        print(f"Could not encode image {image_path}: {e}")
        return None


def train(train_dir: str, model_save_path: str = None, n_neighbors: int = None,
          knn_algo: str = 'ball_tree', verbose: bool = False) -> neighbors.KNeighborsClassifier:
    X = []
    Y = []

    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            try:
                image_encoding_result = image_encoding(img_path)
                if image_encoding_result is None:
                    print(f"Image {img_path} not suitable for training: Didn't find a face")
                else:
                    X.append(image_encoding_result)
                    Y.append(class_dir)
            except Exception as e:
                print(f"Could not read image {img_path}: {e}")

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, Y)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf



def predict(X_img_path, knn_clf=None, model_path=None, allowed_extensions=None,distance_threshold=0.6):
    # 이미지 경로가 유효한지 확인합니다.
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise ValueError("Invalid image path {}".format(X_img_path))

    # 모델이 제공되지 않았다면 파일로부터 불러옵니다.
    if knn_clf is None and model_path is not None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 이미지에서 얼굴을 찾습니다.
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # 얼굴이 없다면 빈 결과를 반환합니다.
    if len(X_face_locations) == 0:
        return []

    # 찾은 얼굴들의 임베딩을 추출합니다.
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # 각 얼굴에 대해 가장 가까운 이웃을 찾아서 예측합니다.
    closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    predictions = [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                   zip(knn_clf.predict(face_encodings), X_face_locations, are_matches)]

    return predictions
