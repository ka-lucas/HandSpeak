import os
import csv
import psycopg2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
from dotenv import load_dotenv
import cv2

# Carrega variáveis de ambiente do .env
load_dotenv()

# Configurações do banco de dados
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")


# Caminho absoluto para o modelo
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/hand_landmarker.task'))
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo não encontrado em: {MODEL_PATH}")

# Inicializa detector MediaPipe
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Caminhos de saída
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'landmarks.csv')
OUTPUT_IMG_DIR = os.path.join(OUTPUT_DIR, 'landmarked_images')
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def fetch_images_and_letters():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT l.letters_numbers AS letter, i.images_url AS image_url
        FROM letter_number_images i
        INNER JOIN letters_numbers l ON i.letter_number_id::text = l.id::text
    """)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

def main():
    print("🚀 Extraindo landmarks e salvando CSV...")

    # Cria CSV e cabeçalho
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
        writer.writerow(header)

        images_and_letters = fetch_images_and_letters()

        for idx, (letter, image_url) in enumerate(images_and_letters):
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code != 200:
                    print(f"❌ Erro ao baixar: {image_url} (status {response.status_code})")
                    continue
            except requests.exceptions.RequestException as e:
                print(f"❌ Timeout/download falhou: {image_url} ({e})")
                continue

            temp_path = os.path.join(OUTPUT_IMG_DIR, f"temp_{letter}_{idx}.jpg")
            with open(temp_path, "wb") as img_file:
                img_file.write(response.content)

            image = cv2.imread(temp_path)
            height, width, _ = image.shape
            size = max(height, width)

            square_image = cv2.copyMakeBorder(
                image,
                (size - height) // 2,
                (size - height + 1) // 2,
                (size - width) // 2,
                (size - width + 1) // 2,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB))
            result = detector.detect(mp_image)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                row = [letter] + [coord for landmark in hand for coord in (landmark.x, landmark.y, landmark.z)]
                writer.writerow(row)

                annotated = square_image.copy()
                for landmark in hand:
                    x_px = int(landmark.x * size)
                    y_px = int(landmark.y * size)
                    cv2.circle(annotated, (x_px, y_px), 4, (0, 255, 0), -1)

                for connection in mp.solutions.hands.HAND_CONNECTIONS:
                    start = hand[connection[0]]
                    end = hand[connection[1]]
                    start_point = (int(start.x * size), int(start.y * size))
                    end_point = (int(end.x * size), int(end.y * size))
                    cv2.line(annotated, start_point, end_point, (255, 0, 0), 2)

                output_img_path = os.path.join(OUTPUT_IMG_DIR, f"{letter}_{idx}_landmarked.jpg")
                cv2.imwrite(output_img_path, annotated)

            else:
                print(f"⚠️ Nenhuma mão detectada: {image_url}")

            os.remove(temp_path)

    print(f"✅ Concluído! CSV salvo em: {OUTPUT_CSV}")
    print(f"🖼️ Imagens anotadas salvas em: {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    main()
