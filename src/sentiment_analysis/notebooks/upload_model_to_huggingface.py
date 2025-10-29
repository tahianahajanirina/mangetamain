import os

from huggingface_hub import HfApi, login

# 1. Se connecter à Hugging Face
print("Connexion à Hugging Face...")
login()

# 2. Définir le chemin de votre modèle (AJUSTEZ CE CHEMIN)
model_path = r"E:\Cours Télécom Paris\IADATA700 - Kik Big Data - Charlotte LACLAU\Projet-groupe\mangetamain\src\sentiment_analysis\model"

# 3. Vérifier que le dossier existe
if not os.path.exists(model_path):
    print(f"❌ ERREUR: Le dossier {model_path} n'existe pas!")
    print("\nContenu du dossier actuel:")
    print(os.listdir("."))
    exit()

print(f"✅ Dossier trouvé: {model_path}")
print(f"Fichiers dans le dossier: {os.listdir(model_path)}")

# 4. Uploader vers Hugging Face
print("\nUpload en cours...")
api = HfApi()
api.upload_folder(
    folder_path=model_path,
    repo_id="TahianaAndriambahoaka/sentiment-analysis-food-reviews",
    repo_type="model",
)

print("✅ Upload terminé avec succès!")
print(
    "Votre modèle est disponible sur: https://huggingface.co/TahianaAndriambahoaka/sentiment-analysis-food-reviews"
)
