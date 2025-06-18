## HandSpeak

# 🤟 HandSpeak - Reconhecimento de Libras com Visão Computacional

Este projeto utiliza **MediaPipe**, **scikit-learn** e **OpenCV** para detectar gestos de Libras capturados via webcam ou imagens, classificando as letras do alfabeto em tempo real.

---

## 📂 Estrutura de Pastas

```
.
├── data/
│   ├── processed/
│   │   ├── landmarked_images/
│   │   │   └── ...
│   │   └── landmarks.csv
│   ├── raw/
│   │   ├── training/
│           └── ...
│    
│   
├── models/
│   └──hand_landmarker.task
|   └──random_forest_model.pkl
|
├── src/
│   ├── prediction/
│   │   ├── predict_image_cam.py
│   │   └── predict_image.py
│   ├── processing/
│   │   └── extract_landmarks_from_db.py
│   ├── training/
│   │   └── train_landmarks_classifier.py
│   ├── utils/
│       └──normalization.py
│
|── tests/
│   ├── upgradeSupaBase.py/
│   ├── ReUpgradeSupaBase.py
│      
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.8+
- OpenCV
- MediaPipe
- scikit-learn
- joblib
- numpy, pandas, matplotlib, seaborn

Instale com:

```bash
pip install -r requirements.txt
```

---

## 🧠 Treinamento do Modelo

```bash
python app/train_landmarks_classifier.py
```

Este script:
- Lê os dados do `landmarks.csv`
- Normaliza os landmarks com base no centro da mão
- Treina 4 modelos (RandomForest, KNN, SVM, MLP)
- Avalia com matriz de confusão
- Salva o melhor modelo em `models/random_forest_model.pkl`

---

## 📸 Predição por Imagem

```bash
python app/predict_image.py
```

- O caminho da imagem é configurado no próprio script.
- Exibe o resultado da predição + landmarks se desejado.

---

## 🎥 Predição por Webcam

```bash
python app/predict_image_cam.py
```

- Reconhece letras de Libras em tempo real.
- Monta frases com letras estáveis.
- Reseta a frase após 1 minuto de inatividade.

---

## 🐳 Executando com Docker

### 1. Build da imagem

```bash
docker build -t libras-predictor .
```

### 2. Executar com acesso à webcam

```bash
docker run --rm -it   --device=/dev/video0   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   libras-predictor
```

> 💡 Em sistemas Linux, execute `xhost +local:root` para liberar o uso da webcam via Docker.  
> Em Windows/Mac, o uso de webcam com Docker pode exigir configurações adicionais (ex: WSL2 + Docker Desktop).

---

## ✨ Créditos

Projeto desenvolvido por Felipe Vasconcellos e Katarine Melo para reconhecimento de Libras com foco em acessibilidade e tecnologia assistiva.
