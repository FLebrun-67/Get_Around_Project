# Guide de Deploiement Hugging Face

## Fichiers Generes

### Dossier hf_deployment/api/
- `trained_model.pkl` - Modele MLflow exporte (Run: 7da1f983c7c34ae1a3c4f1f82e15ee7e)
- `model_metadata.json` - Metadonnees completes
- `run_id.txt` - Reference MLflow
- `README.md` - Documentation API
- `requirements.txt` - Dependances Python

### Dossier hf_deployment/dashboard/
- `README.md` - Documentation Dashboard
- `requirements.txt` - Dependances Streamlit

## Etapes de Deploiement

### 1. Creer le Space API
1. Aller sur https://huggingface.co/new-space
2. Nom: `getaround-api` (ou autre nom)
3. SDK: `Gradio` 
4. Hardware: `CPU basic` (gratuit)
5. Upload tous les fichiers de `api/`
6. Ajouter le fichier `app.py` (version HF adaptee de main2.py)

### 2. Creer le Space Dashboard  
1. Aller sur https://huggingface.co/new-space
2. Nom: `getaround-dashboard` (ou autre nom)
3. SDK: `Streamlit`
4. Hardware: `CPU basic` (gratuit)
5. Upload tous les fichiers de `dashboard/`
6. Ajouter le fichier `streamlit_app.py` (version HF adaptee de app.py)
7. **IMPORTANT:** Modifier l'URL API dans streamlit_app.py

### 3. Connecter les Deux Spaces
Dans `streamlit_app.py`, ligne ~15:
```python
API_URL = "https://ton-username-getaround-api.hf.space/predict"
```
Remplace `ton-username` par ton vrai nom d'utilisateur HF.

### 4. Test Final
- API Docs: https://ton-username-getaround-api.hf.space/docs
- Dashboard: https://ton-username-getaround-dashboard.hf.space

## Informations du Modele
- Run ID MLflow: 7da1f983c7c34ae1a3c4f1f82e15ee7e
- Performance R2: 0.7500
- Date export: 2025-06-18

## Architecture Complete
- **Local:** Docker Compose + MLflow (pour developpement)
- **Cloud:** 2 HF Spaces + modele exporte (pour production)

## Backup Local
Garde ton architecture Docker Compose locale pour:
- Developpement et experimentation
- Demonstration complete avec MLflow UI
- Formation de nouveaux modeles
