import cv2
import mediapipe as mp
import numpy as np
import joblib  # para carregar modelo
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. Carregar modelo e scaler
# ---------------------------
model = joblib.load("random_forest_model.pkl")  # ou outro nome de arquivo
scaler = joblib.load("scaler.pkl")

# ---------------------------
# Função para redimensionar imagem para quadrado
# ---------------------------
def resize_to_square(image):
    size = max(image.shape[:2])
    new_image = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - image.shape[0]) // 2
    x_offset = (size - image.shape[1]) // 2
    new_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
    return new_image


# ---------------------------
# 2. Função para extrair landmarks e visualizar a área da mão
# ---------------------------
def extract_hand_landmarks(image_path, visualize=False):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionar imagem para quadrado antes de processar
    image_rgb_square = resize_to_square(image_rgb)

    results = hands.process(image_rgb_square)

    if not results.multi_hand_landmarks:
        print("❌ Nenhuma mão detectada.")
        return None

    landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])  # agora incluindo o z

    
    # Visualizar ou salvar a imagem com landmarks desenhados
    if visualize:
        annotated_image = cv2.cvtColor(image_rgb_square, cv2.COLOR_RGB2BGR).copy()
        mp_drawing.draw_landmarks(annotated_image, landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Área da Mão Detectada", annotated_image)  # Exibe a imagem
        cv2.waitKey(0)  # Aguarda uma tecla para fechar a janela
        cv2.destroyAllWindows()

    return np.array(coords)

# ---------------------------
# 3. Fazer predição a partir da imagem
# ---------------------------
def predict_from_image(image_path):
    features = extract_hand_landmarks(image_path, visualize= False)
    if features is None:
        return "❌ Mão não detectada."

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

# ---------------------------
# 4. Exemplo de uso
# ---------------------------
if __name__ == "__main__":
    image_path = "C:/Project_HandSpeak/letraC.png"  # coloque o caminho da sua imagem aqui
    letra_predita = predict_from_image(image_path)
    print(f"Letra preditada: {letra_predita}")
    extract_hand_landmarks(image_path, visualize=True)
