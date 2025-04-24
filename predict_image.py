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
# 2. Função para extrair landmarks
# ---------------------------
def extract_hand_landmarks(image_path):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        print("❌ Nenhuma mão detectada.")
        return None

    landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])  # agora incluindo o z

    return np.array(coords)

# ---------------------------
# 3. Fazer predição a partir da imagem
# ---------------------------
def predict_from_image(image_path):
    features = extract_hand_landmarks(image_path)
    if features is None:
        return "Mão não detectada."

    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

# ---------------------------
# 4. Exemplo de uso
# ---------------------------
if __name__ == "__main__":
    image_path = "Y.png"  # coloque o caminho da sua imagem aqui
    letra_predita = predict_from_image(image_path)
    print(f"🅰️ Letra predita: {letra_predita}")
