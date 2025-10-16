# Dockerfile pour le projet MangeTaMain
FROM python:3.11-slim

# Informations sur l'image
LABEL maintainer="mangetamain-project"
LABEL description="Container pour le système de recommandation de recettes Food.com"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Créer un utilisateur non-root pour la sécurité
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Copier les fichiers de configuration Python
COPY --chown=app:app pyproject.toml ./

# Installer les dépendances Python
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --user --no-cache-dir .

# Copier le code source
COPY --chown=app:app . .

# Créer les dossiers nécessaires
RUN mkdir -p data/raw data/processed outputs/figures outputs/models outputs/reports

# Port pour d'éventuels services web
EXPOSE 8000

# Variables d'environnement pour Python
ENV PATH="/home/app/.local/bin:${PATH}"
ENV PYTHONPATH="/home/app:${PYTHONPATH}"

# Container reste en vie pour développement VS Code
CMD ["tail", "-f", "/dev/null"]