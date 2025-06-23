# 🚗 GetAround Price Prediction Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![MLflow](https://img.shields.io/badge/MLflow-2.19-orange)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

**Prédiction des prix de location de véhicules** avec une architecture complète de Machine Learning en production.

---

## 📋 **Table des Matières**

- [🎯 Aperçu du Projet](#-aperçu-du-projet)
- [🏗️ Architecture](#️-architecture)
- [🚀 Déploiement](#-déploiement)
- [🔧 Installation Locale](#-installation-locale)
- [📊 Utilisation](#-utilisation)
- [🎭 Démonstration](#-démonstration)
- [📈 Performances](#-performances)
- [🤝 Contributeurs](#-contributeurs)

---

## 🎯 **Aperçu du Projet**

### **Problématique**
GetAround, leader de l'autopartage, doit optimiser les prix de location de ses véhicules pour maximiser les revenus tout en restant compétitif.

### **Solution**
Développement d'une **API de prédiction de prix** basée sur l'Intelligence Artificielle, capable de recommander des tarifs optimaux en fonction des caractéristiques des véhicules.

### **Technologies Utilisées**
- **🤖 Machine Learning** : XGBoost, scikit-learn
- **🚀 API** : FastAPI avec validation Pydantic
- **📊 Interface** : Streamlit pour l'interface utilisateur
- **🔬 Monitoring** : MLflow pour le suivi des expériences
- **🐳 Déploiement** : Docker + Hugging Face Spaces

---

## 🏗️ **Architecture**

### **🎯 Architecture Hybride : Développement Local + Production Cloud**

Notre projet implémente une **architecture hybride** professionnelle :

```
📊 DÉVELOPPEMENT LOCAL (MLOps Complet)
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🤖 Trainer    │    │   📊 MLflow     │    │   🚀 API        │    │ 📊 Dashboard    │
│   (XGBoost)     │────│   (Tracking)    │────│   (FastAPI)     │────│  (Streamlit)    │
│   Port: N/A     │    │   Port: 5000    │    │   Port: 8000    │    │   Port: 8501    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘

🌐 PRODUCTION CLOUD (Déploiement Simplifié)
┌─────────────────────────────────┐    ┌─────────────────────────────────┐
│  🚀 HF Space #1 - API          │    │  📊 HF Space #2 - Dashboard     │
│  • FastAPI + Modèle exporté    │◄───┤  • Streamlit                    │
│  • Monitoring MLflow léger     │    │  • Interface utilisateur        │
│  • Documentation auto          │    │  • Connexion API REST           │
└─────────────────────────────────┘    └─────────────────────────────────┘
```

---

## 🚀 **Déploiement**

### **🌐 Production (Hugging Face Spaces)**

**URLs de Démonstration :**
- 🚀 **API** : [https://beltzark-getaround-api.hf.space](https://beltzark-getaround-api.hf.space)
- 📊 **Dashboard** : [https://beltzark-getaround-dashboard.hf.space](https://beltzark-getaround-dashboard.hf.space)

#### **Avantages du Déploiement HF :**
- ✅ **Accessibilité Publique** : URLs permanentes pour démonstration
- ✅ **Scalabilité** : Infrastructure gérée par Hugging Face
- ✅ **Séparation des Concerns** : API et Frontend indépendants
- ✅ **Facilité de Maintenance** : Déploiement simplifié

#### **Architecture de Production :**
```
Internet ──► HF Space API ──► Modèle ML ──► Prédiction
              │
              └──► HF Space Dashboard ──► Interface Utilisateur
```

### **🏠 Développement Local (Docker Compose)**

#### **Pourquoi Docker Compose ?**

**Docker Compose** est utilisé pour l'environnement de développement car il offre :

1. **🔬 Orchestration Complète**
   - **4 services** démarrés simultanément d'un seul clic
   - **Communication inter-services** automatique
   - **Volumes partagés** pour persistance des données

2. **🧪 Environnement de Développement Idéal**
   - **MLflow UI** pour explorer les expériences (port 5000)
   - **API complète** avec rechargement automatique (port 8000)
   - **Dashboard interactif** connecté en temps réel (port 8501)
   - **Trainer** pour re-entraîner les modèles à la demande

3. **🎯 Facilite le Workflow MLOps**
   - **Expérimentation** : Tester rapidement de nouveaux modèles
   - **Comparaison** : MLflow UI pour comparer les performances
   - **Validation** : Interface complète pour tester l'API
   - **Export** : Pipeline d'export vers production HF

4. **🔒 Reproductibilité**
   - **Environnements isolés** : Chaque service dans son conteneur
   - **Versions fixes** : Dependencies verrouillées
   - **Configuration centralisée** : docker-compose.yml

#### **Services Docker :**
```yaml
services:
  trainer:    # Entraîne le modèle XGBoost + MLflow
  mlflow:     # Interface de tracking des expériences  
  api:        # FastAPI avec modèle chargé depuis MLflow
  dashboard:  # Streamlit connecté à l'API locale
```

---

## 🔧 **Installation Locale**

### **Prérequis**
- Python 3.11+
- Docker Desktop
- Git

### **🐳 Démarrage Rapide (Recommandé)**

```bash
# 1. Cloner le projet
git clone https://github.com/FLebrun67/getaround-project
cd getaround-project

# 2. Démarrer l'architecture complète
docker-compose up --build

# 3. Accéder aux services
# MLflow UI:    http://localhost:5000
# API:          http://localhost:8000/docs  
# Dashboard:    http://localhost:8501
```

### **⚡ Démarrage Manuel (Développement)**

```bash
# 1. Installation des dépendances
pip install -r requirements.api.txt
pip install -r requirements.streamlit.txt

# 2. Entraînement du modèle
python train_model.py

# 3. Démarrage des services
# Terminal 1: MLflow
mlflow ui --port 5000

# Terminal 2: API
cd api
python main2.py

# Terminal 3: Dashboard  
cd streamlit
streamlit run app.py
```

### **📤 Export vers Production**

```bash
# Exporter le modèle pour Hugging Face
python export_for_hf.py

# Fichiers générés dans hf_deployment/
├── api/
│   ├── app.py              # Version HF de l'API
│   ├── trained_model.pkl   # Modèle exporté
│   └── requirements.txt
└── dashboard/
    ├── streamlit_app.py    # Version HF du dashboard
    └── requirements.txt
```

---

## 📊 **Utilisation**

### **🎯 Prédiction de Prix**

#### **Via l'Interface Web :**
1. Aller sur [Dashboard](https://beltzark-getaround-dashboard.hf.space)
2. Configurer les paramètres du véhicule dans la sidebar
3. Cliquer sur "Prédire le prix"

#### **Via l'API REST :**
```bash
curl -X POST "https://beltzark-getaround-api.hf.space/predict" \
     -H "Content-Type: application/json" \
     -d '{
         "model_key": "Renault",
         "mileage": 50000,
         "engine_power": 120,
         "fuel": "diesel",
         "paint_color": "black",
         "car_type": "sedan",
         "private_parking_available": true,
         "has_gps": true,
         "has_air_conditioning": true,
         "automatic_car": false,
         "has_getaround_connect": true,
         "has_speed_regulator": true,
         "winter_tires": false
     }'
```

#### **Réponse :**
```json
{
  "rental_price": 142.67,
  "currency": "EUR",
  "period": "per_day",
  "status": "success",
  "model_confidence": "high"
}
```

### **📈 Monitoring & Analytics**

#### **Métriques du Modèle :**
- **R²** : 0.7500 (75% de variance expliquée)
- **RMSE** : 16.23€ (erreur quadratique moyenne)
- **MAE** : 10.34€ (erreur absolue moyenne)

#### **Monitoring en Production :**
- **MLflow Stats** : `/mlflow-stats` endpoint
- **Santé de l'API** : `/health` endpoint
- **Informations Modèle** : `/model-info` endpoint

---

## 🎭 **Démonstration**

### **🎯 Workflow de Démonstration pour le Jury**

#### **1. 🏠 Architecture Locale Complète (5 minutes)**
```bash
docker-compose up --build
```

**Points à montrer :**
- **MLflow UI** (http://localhost:5000) : Historique des expériences
- **API FastAPI** (http://localhost:8000/docs) : Documentation interactive
- **Dashboard Streamlit** (http://localhost:8501) : Interface utilisateur
- **Workflow MLOps** : Train → Track → Deploy


**URLs de l'API et du Dashboard**
- [API Production](https://beltzark-getaround-api.hf.space/docs)
- [Dashboard Production](https://beltzark-getaround-dashboard.hf.space)


## 📈 **Performances**

### **📊 Métriques du Modèle**

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **R²** | 0.7500 | 75% de la variance des prix expliquée |
| **RMSE** | 16.23€ | Erreur moyenne de ±16€ |
| **MAE** | 10.34€ | Erreur absolue moyenne de ±10€ |

---

## 🛠️ **Structure du Projet**

```
📁 GetAround_project/
├── 📁 api/                    # API FastAPI
│   ├── 📄 app.py           # API locale avec MLflow
│   └── 📄 requirements.txt
├── 📁 streamlit/             # Dashboard Streamlit  
│   ├── 📄 streamlit_app.py             # Dashboard local
│   └── 📄 requirements.txt
├── 📁 data/                  # Données d'entraînement
│   ├── 📄 get_around_pricing_project.csv
    └── 📄 get_around_delay_analysis_cleaned.csv
├── 📁 hf_deployment/         # Déploiement Hugging Face
│   ├── 📁 api/               # API pour HF Spaces
│   │   ├── 📄 app.py         # Version HF de l'API
│   │   ├── 📄 trained_model.pkl
│   │   └── 📄 requirements.txt
│   └── 📁 dashboard/         # Dashboard pour HF Spaces
│       ├── 📄 streamlit_app.py
│       └── 📄 requirements.txt
    ├── 📄 get_around_pricing.csv
    └── 📄 get_around_delay_analysis_cleaned.csv
├── 📁 mlruns/                # Expériences MLflow
├── 📄 train_model.py         # Entraînement du modèle
├── 📄 export_for_hf.py       # Export vers HF
├── 📄 docker-compose.yml     # Orchestration locale
├── 📄 dockerfile.api
├── 📄 dockerfile.streamlit
├── 📄 dockerfile.trainer
├── 📄 requirements.mlflow.txt
├── 📄 requirements.api.txt
├── 📄 requirements.streamlit.txt
└── 📄 README.md              # Ce fichier
```

---

**Florent LEBRUN** - Data Scientist  
📧 Email: [flebrun67@gmail.com]  
🔗 LinkedIn: [www.linkedin.com/in/f-lebrun1989]  
🐙 GitHub: [FLebrun67](https://github.com/FLebrun67)  

---

## 🆘 **Support**

### **🐛 Problèmes Courants**

**Docker ne démarre pas :**
```bash
# Vérifier Docker Desktop
docker --version
docker ps
```

**API non accessible :**
```bash
# Vérifier les logs
docker-compose logs api
```

**Modèle non trouvé :**
```bash
# Re-entraîner le modèle
python train_model.py
```

---

*Développé avec ❤️ pour GetAround - Projet de formation Data Science 2025*