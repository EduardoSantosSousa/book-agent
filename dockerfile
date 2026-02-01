# Usar imagem Python oficial com CUDA (mais confiável)
FROM python:3.11-slim

# Instalar dependências do sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements
COPY requirements-prod.txt .

# 1️⃣ ATUALIZAR pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2️⃣ INSTALAR dependências básicas
RUN pip install --no-cache-dir \
    numpy==1.24.4 \
    scipy==1.13.0 \
    pandas==2.2.3

# 3️⃣ INSTALAR PyTorch (CPU para teste, depois adicionamos GPU)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# 4️⃣ INSTALAR outras dependências
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copiar código
COPY . .

# Criar diretório para credenciais
RUN mkdir -p /app/credentials

# Variáveis de ambiente
ENV CUDA_VISIBLE_DEVICES=0
ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/book-agent-api-3156cc932afc.json

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "5", "--timeout", "600", "app:app"]