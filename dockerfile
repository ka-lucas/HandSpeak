# Base Python com suporte a OpenCV
FROM python:3.10-slim

# Variáveis de ambiente para UTF-8
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Diretório de trabalho
WORKDIR /app

# Copia os arquivos para dentro do container
COPY . /app

# Instala dependências do sistema para OpenCV, MediaPipe e PostgreSQL
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["python", "src/predictors/predict_image_cam.py"]
