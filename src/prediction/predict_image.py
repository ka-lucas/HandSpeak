import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Adiciona o caminho para importar de src/utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.normalization import normalize_landmarks_center

# ======================
# CONSTANTES DE ERRO
# ======================
ERROR_CODES = {
    "MODEL_LOAD": "ER001",
    "IMAGE_LOAD": "ER002",
    "LANDMARK_EXTRACTION": "ER003",
    "PREDICTION": "ER004",
    "NO_HAND": "ER005",
    "VISUALIZATION": "ER006"
}

# ======================
# 1. CARREGAMENTO DO MODELO
# ======================
def load_model():
    try:
        model = joblib.load("models/random_forest_model.pkl")
        return model
    except Exception as e:
        print(f"❌ [{ERROR_CODES['MODEL_LOAD']}] Falha ao carregar modelo - {type(e).__name__}: {str(e)}")
        exit()

model = load_model()

# ======================
# 2. FUNÇÕES DE PROCESSAMENTO
# ======================
def resize_to_square(image):
    try:
        size = max(image.shape[:2])
        new_image = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - image.shape[0]) // 2
        x_offset = (size - image.shape[1]) // 2
        new_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        return new_image
    except Exception as e:
        print(f"❌ [{ERROR_CODES['IMAGE_LOAD']}] Falha no redimensionamento - {type(e).__name__}: {str(e)}")
        return None

def extract_hand_landmarks(image_path, visualize=False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ [{ERROR_CODES['IMAGE_LOAD']}] Imagem não encontrada")
            return None
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_square = resize_to_square(image_rgb)
        results = hands.process(image_rgb_square)

        if not results.multi_hand_landmarks:
            if visualize:
                print(f"⚠️ [{ERROR_CODES['NO_HAND']}] Nenhuma mão detectada")
            return None

        landmarks = results.multi_hand_landmarks[0]
        coords = np.array([coord for lm in landmarks.landmark for coord in [lm.x, lm.y, lm.z]])

        if visualize:
            try:
                annotated_image = cv2.cvtColor(image_rgb_square, cv2.COLOR_RGB2BGR).copy()
                mp_drawing.draw_landmarks(annotated_image, landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Detecção de Mão", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"⚠️ [{ERROR_CODES['VISUALIZATION']}] Erro na visualização - {type(e).__name__}: {str(e)}")

        return coords

    except Exception as e:
        print(f"❌ [{ERROR_CODES['LANDMARK_EXTRACTION']}] Erro no processamento - {type(e).__name__}: {str(e)}")
        return None

# ======================
# 3. FUNÇÃO DE PREDIÇÃO
# ======================
def predict_from_image(image_path):
    features = extract_hand_landmarks(image_path, visualize=False)
    if features is None:
        return None

    try:
        normalized = normalize_landmarks_center(features)
        prediction = model.predict([normalized])
        return prediction[0]
    except Exception as e:
        print(f"❌ [{ERROR_CODES['PREDICTION']}] Erro na predição - {type(e).__name__}: {str(e)}")
        return None

# ======================
# 4. EXECUÇÃO PRINCIPAL
# ======================
if __name__ == "__main__":
    image_path = "data/training/A/1002.png"  # ajuste para o seu caminho real

    print("\n=== ANÁLISE DE IMAGEM - LIBRAS ===")
    print(f"🖼️ Imagem: {image_path}\n")

    try:
        features = extract_hand_landmarks(image_path, visualize=False)

        if features is None:
            print(f"❌ [{ERROR_CODES['NO_HAND']}] Nenhuma mão detectada")
        else:
            prediction = predict_from_image(image_path)
            if prediction:
                print(f"✅ Resultado: Letra '{prediction}' identificada")
            else:
                print(f"❌ [{ERROR_CODES['PREDICTION']}] A mão foi detectada, mas a predição falhou")

            input("\n🔍 Pressione Enter para visualizar a imagem detectada...")
            extract_hand_landmarks(image_path, visualize=True)

    except KeyboardInterrupt:
        print("\n⏹ Interrompido pelo usuário")
    except Exception as e:
        print(f"❌ [ER000] Erro inesperado - {type(e).__name__}: {str(e)}")

    print("\n=== FIM ===")
