import os
import sys
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

# Adiciona o caminho do diretório src/utils para importações
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.normalization import normalize_landmarks_center


# CONSTANTES DE ERROS DETECTADAS
ERROR_CODES = {
    "MODEL_LOAD": "ER001",
    "LANDMARK_EXTRACTION": "ER003",
    "PREDICTION": "ER004",
    "NO_HAND": "ER005",
    "VISUALIZATION": "ER006"
}

# CORES POR LETRA
LETTER_COLORS = {
    'A': (0, 0, 255), 'B': (0, 255, 0), 'C': (255, 0, 0), 'D': (255, 255, 0),
    'E': (255, 0, 255), 'F': (0, 255, 255), 'G': (128, 0, 128), 'H': (255, 165, 0),
    'I': (0, 128, 255), 'J': (128, 128, 0), 'K': (255, 20, 147), 'L': (0, 206, 209),
    'M': (124, 252, 0), 'N': (138, 43, 226), 'O': (240, 230, 140), 'P': (255, 105, 180),
    'Q': (139, 69, 19), 'R': (47, 79, 79), 'S': (255, 215, 0), 'T': (70, 130, 180),
    'U': (154, 205, 50), 'V': (199, 21, 133), 'W': (210, 105, 30), 'X': (255, 69, 0),
    'Y': (123, 104, 238), 'Z': (32, 178, 170)
}

# Carregar modelo
def load_model():
    try:
        model_path = os.path.join("models", "random_forest_model.pkl")
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"❌ [{ERROR_CODES['MODEL_LOAD']}] Falha ao carregar modelo - {type(e).__name__}: {str(e)}")
        exit()

model = load_model()

# Execução Principal com Webcam
if __name__ == "__main__":
    print("\n=== MODO WEBCAM - Análise de Libras ===\n")
    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    frase = ""
    letra_anterior = ""
    contador = 0
    MAX_FRAMES_MESMA_LETRA = 40

    tempo_ultima_letra_detectada = time.time()
    TEMPO_LIMITE_ZERAR = 60  # 1 minuto

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Erro ao capturar frame")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            letra = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    try:
                        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        features = normalize_landmarks_center(coords)

                        probs = model.predict_proba([features])[0]
                        predicted_index = np.argmax(probs)
                        letra = model.classes_[predicted_index]
                        confidence = probs[predicted_index] * 100

                        tempo_ultima_letra_detectada = time.time()

                        if letra == letra_anterior:
                            contador += 1
                        else:
                            contador = 1
                            letra_anterior = letra

                        # Se estabilizado, adiciona à frase
                        if contador == MAX_FRAMES_MESMA_LETRA:
                            frase += letra
                            print(f"📝 Frase atual: {frase}")
                            contador = 0

                        cor = LETTER_COLORS.get(letra, (255, 255, 255))
                        cv2.putText(
                            frame,
                            f"Letra: {letra} ({confidence:.1f}%)",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            cor,
                            3
                        )

                    except Exception as e:
                        print(f"❌ [{ERROR_CODES['PREDICTION']}] Predição falhou - {type(e).__name__}: {str(e)}")

            tempo_atual = time.time()
            if tempo_atual - tempo_ultima_letra_detectada >= TEMPO_LIMITE_ZERAR:
                if frase:
                    frase = ""
                    print("🧹 Frase zerada por inatividade.")
                tempo_ultima_letra_detectada = tempo_atual

            # Mostra frase atual
            cv2.putText(
                frame,
                f"Frase: {frase}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2
            )

            # Instrução na tela
            cv2.putText(
                frame,
                "Tecla 'q'= sair | 'Backspace' ou 'd'= apagar letra",
                (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2
            )

            cv2.imshow("HandSpeak Alphabet - Webcam", frame)

            # Captura tecla pressionada
            key = cv2.waitKey(1) & 0xFF

            # Se pressionar Backspace (8) ou 'd' (100), remove última letra
            if key in [8, ord('d')]:
                if frase:
                    frase = frase[:-1]
                    print(f"📝 Última letra removida. Frase atual: {frase}")

            # Se pressionar 'q', encerra o programa
            if key == ord('q'):
                break
            
    except KeyboardInterrupt:
        print("\n⏹ Encerrado pelo usuário.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== FIM ===")
