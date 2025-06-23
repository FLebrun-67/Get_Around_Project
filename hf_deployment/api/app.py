# app.py - Version Hugging Face adaptée de main2.py
# 🚀 À placer dans hf_deployment/api/

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal
import mlflow
import mlflow.sklearn
import pickle
import tempfile
import json
from pathlib import Path
from datetime import datetime
import time

# 🔬 Configuration MLflow léger pour HF
def setup_mlflow_hf():
    """
    Configure MLflow pour fonctionner sur Hugging Face Spaces
    Version légère avec stockage temporaire
    """
    # Dossier temporaire pour MLflow sur HF
    mlflow_dir = Path(tempfile.gettempdir()) / "mlruns"
    mlflow_dir.mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri(f"file://{mlflow_dir}")
    print(f"🔬 MLflow configuré : {mlflow_dir}")
    
    # Créer une expérience pour la production HF
    try:
        mlflow.create_experiment("hf_production_monitoring")
        print("✅ Expérience créée : hf_production_monitoring")
    except mlflow.exceptions.MlflowException:
        # Expérience existe déjà
        print("✅ Expérience hf_production_monitoring trouvée")
        pass
    
    mlflow.set_experiment("hf_production_monitoring")
    return str(mlflow_dir)

# 🔄 Fonction hybride pour charger le modèle (pickle prioritaire + MLflow fallback)
def load_model_intelligent():
    """
    Chargement intelligent du modèle :
    1. Priorité : pickle exporté depuis MLflow
    2. Fallback : run_id MLflow si disponible  
    3. Informations : métadonnées du modèle
    """
    print("📥 Chargement du modèle...")
    
    # Option 1 : Charger depuis pickle (PRIORITÉ pour HF)
    model_path = Path("trained_model.pkl")
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Modèle chargé depuis pickle : {model_path}")
            
            # Charger les métadonnées si disponibles
            metadata = load_model_metadata()
            return model, "pickle", metadata
            
        except Exception as e:
            print(f"⚠️ Erreur pickle : {e}")
    
    # Option 2 : Fallback MLflow si run_id disponible
    run_id_file = Path("run_id.txt")
    if run_id_file.exists():
        try:
            with open(run_id_file, 'r') as f:
                run_id = f.read().strip()
            
            print(f"📋 Run ID trouvé : {run_id}")
            
            # Essayer de charger depuis MLflow (si tracking URI accessible)
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Modèle chargé depuis MLflow : {run_id}")
            
            metadata = {"run_id": run_id, "source": "mlflow_fallback"}
            return model, "mlflow", metadata
            
        except Exception as e:
            print(f"⚠️ MLflow fallback échoué : {e}")
    
    print("❌ Aucune méthode de chargement disponible")
    return None, "none", {}

def load_model_metadata():
    """
    Charge les métadonnées du modèle exporté
    """
    metadata_path = Path("model_metadata.json")
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"📋 Métadonnées chargées : Run {metadata.get('run_id', 'Unknown')}")
            return metadata
        except Exception as e:
            print(f"⚠️ Erreur métadonnées : {e}")
    
    return {}

# 🔬 Fonction de logging des prédictions (IDENTIQUE à la version complète)
def log_prediction_to_mlflow(input_data, prediction, confidence, processing_time=None):
    """
    Log chaque prédiction dans MLflow pour monitoring en production
    Cette fonction ne fait pas échouer l'API si MLflow a un problème
    """
    try:
        with mlflow.start_run(run_name=f"prediction_{datetime.now().strftime('%H%M%S')}"):
            # 📊 Log des paramètres d'entrée
            mlflow.log_params(input_data)
            
            # 📈 Log des métriques de prédiction
            mlflow.log_metric("predicted_price", prediction)
            
            # 📊 Log du niveau de confiance
            confidence_scores = {"high": 1.0, "medium": 0.5, "low": 0.1}
            mlflow.log_metric("confidence_score", confidence_scores.get(confidence, 0.0))
            
            # ⏱️ Log du temps de traitement
            if processing_time:
                mlflow.log_metric("processing_time_ms", processing_time)
            
            # 🏷️ Log de métadonnées
            mlflow.log_param("deployment_env", "huggingface_spaces")
            mlflow.log_param("model_version", "hf_production")
            mlflow.log_param("timestamp", datetime.now().isoformat())
            
            # 🎯 Tags pour recherche
            mlflow.set_tag("type", "production_prediction")
            mlflow.set_tag("fuel_type", input_data.get("fuel", "unknown"))
            mlflow.set_tag("brand", input_data.get("model_key", "unknown"))
            
            print(f"📊 Prédiction loggée dans MLflow : {prediction}€")
            
    except Exception as e:
        # ⚠️ Logging non critique - ne fait pas échouer la prédiction
        print(f"⚠️ Erreur logging MLflow (non critique) : {e}")
        pass

# ✅ GARDÉES IDENTIQUES : Tes classes Pydantic restent exactement pareilles
class CarFeatures(BaseModel):
    """
    Modèle de données pour les caractéristiques d'un véhicule
    """
    model_key: str = Field(
        ..., 
        description="Marque du véhicule",
        pattern=r'^(Citroën|Peugeot|PGO|Renault|Audi|BMW|Ford|Mercedes|Opel|Porsche|Volkswagen|KIA Motors|Alfa Romeo|Ferrari|Fiat|Lamborghini|Maserati|Lexus|Honda|Mazda|Mini|Mitsubishi|Nissan|SEAT|Subaru|Suzuki|Toyota|Yamaha)$'
    )
    mileage: int = Field(
        ge=0, 
        le=500000, 
        description="Kilométrage du véhicule en km"
    )
    engine_power: int = Field(
        ge=0, 
        le=500, 
        description="Puissance du moteur en chevaux"
    )
    fuel: Literal['diesel', 'petrol', 'hybrid_petrol', 'electro'] = Field(
        description="Type de carburant"
    )
    paint_color: Literal['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange', 'beige', 'brown', 'green'] = Field(
        description="Couleur de la peinture"
    )
    car_type: Literal['convertible', 'coupe', 'estate', 'hatchback', 'sedan', 'subcompact'] = Field(
        description="Type de véhicule"
    )
    private_parking_available: bool = Field(
        description="Parking privé disponible"
    )
    has_gps: bool = Field(
        description="GPS intégré"
    )
    has_air_conditioning: bool = Field(
        description="Climatisation"
    )
    automatic_car: bool = Field(
        description="Transmission automatique"
    )
    has_getaround_connect: bool = Field(
        description="Système GetAround Connect"
    )
    has_speed_regulator: bool = Field(
        description="Régulateur de vitesse"
    )
    winter_tires: bool = Field(
        description="Pneus hiver"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_key": "Renault",
                "mileage": 50000,
                "engine_power": 120,
                "fuel": "diesel",
                "paint_color": "black",
                "car_type": "sedan",
                "private_parking_available": True,
                "has_gps": True,
                "has_air_conditioning": True,
                "automatic_car": False,
                "has_getaround_connect": True,
                "has_speed_regulator": True,
                "winter_tires": False
            }
        }}

# ✅ GARDÉE IDENTIQUE : Ta classe de réponse
class PricePrediction(BaseModel):
    """
    Modèle de réponse pour la prédiction de prix
    """
    rental_price: float = Field(description="Prix de location prédit en euros par jour")
    currency: str = Field(default="EUR", description="Devise")
    period: str = Field(default="per_day", description="Période de location")
    status: str = Field(default="success", description="Statut de la prédiction")
    model_confidence: str = Field(description="Niveau de confiance du modèle")

# Variables globales pour stocker le modèle et ses infos
loaded_model = None
model_source = None
model_metadata = {}
mlflow_dir = None

# 🔄 Lifespan adapté pour le chargement hybride
@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model, model_source, model_metadata, mlflow_dir
    print("🚀 Démarrage de l'API GetAround sur Hugging Face...")

    # 🔬 Configuration MLflow léger
    mlflow_dir = setup_mlflow_hf()
    
    # 📥 Chargement intelligent du modèle
    loaded_model, model_source, model_metadata = load_model_intelligent()
    
    if loaded_model:
        print(f"✅ Modèle chargé : {type(loaded_model).__name__}")
        print(f"📊 Source : {model_source}")
        
        if model_metadata:
            run_id = model_metadata.get('run_id', 'Unknown')
            r2_score = model_metadata.get('metrics', {}).get('R2', 'N/A')
            print(f"📈 Performance : R² = {r2_score}")
            print(f"🔗 MLflow Run : {run_id}")
        
        # 🎯 Log du démarrage dans MLflow
        try:
            with mlflow.start_run(run_name="api_startup"):
                mlflow.log_param("startup_time", datetime.now().isoformat())
                mlflow.log_param("model_type", type(loaded_model).__name__)
                mlflow.log_param("model_source", model_source)
                mlflow.log_param("deployment", "huggingface_spaces")
                if model_metadata.get('run_id'):
                    mlflow.log_param("original_run_id", model_metadata['run_id'])
                mlflow.set_tag("event", "api_startup")
                print("📊 Démarrage API loggé dans MLflow")
        except:  # noqa: E722
            pass
    else:
        print("❌ Échec du chargement du modèle.")

    yield
    print("🛑 Arrêt de l'API GetAround")

# ✅ Configuration FastAPI (mise à jour pour HF)
app = FastAPI(
    title="🚗 GetAround Rental Pricing API",
    description="""
    **Bienvenue sur l'API de prédiction des prix GetAround !** 🚗
    
    Cette API vous permet de prédire le prix de location journalier recommandé pour un véhicule
    en fonction de ses caractéristiques.
    
    ## 🚀 **Déployé sur Hugging Face Spaces**
    
    Cette API utilise un modèle de Machine Learning entraîné avec MLflow et exporté pour 
    un déploiement cloud optimisé.
    
    ## 🔬 **Monitoring MLflow Intégré**
    
    - 📊 **Tracking** : Chaque prédiction est enregistrée
    - 📈 **Métriques** : Statistiques temps réel sur `/mlflow-stats`
    - 🎯 **Observabilité** : Monitoring de production complet
    
    ## 📋 Valeurs acceptées pour les champs :
    
    * **model_key**: Citroën, Peugeot, PGO, Renault, Audi, BMW, Ford, Mercedes, Opel, Porsche, 
      Volkswagen, KIA Motors, Alfa Romeo, Ferrari, Fiat, Lamborghini, Maserati, Lexus, Honda, 
      Mazda, Mini, Mitsubishi, Nissan, SEAT, Subaru, Suzuki, Toyota, Yamaha
    * **fuel**: diesel, petrol, hybrid_petrol, electro
    * **paint_color**: black, grey, white, red, silver, blue, orange, beige, brown, green
    * **car_type**: convertible, coupe, estate, hatchback, sedan, subcompact
    * **mileage**: 0 à 500,000 km
    * **engine_power**: 0 à 500 ch
    
    Les autres champs sont des **booléens** (true/false) indiquant la présence ou non des options.
    
    ## 🧪 Exemple d'utilisation :
    ```json
    {
    "automatic_car": false,
    "car_type": "sedan",
    "engine_power": 120,
    "fuel": "diesel",
    "has_air_conditioning": true,
    "has_getaround_connect": true,
    "has_gps": true,
    "has_speed_regulator": true,
    "mileage": 50000,
    "model_key": "Renault",
    "paint_color": "black",
    "private_parking_available": true,
    "winter_tires": false
    }
    ```
    
    ## 🌐 Architecture
    - 🎯 **API Backend** : FastAPI sur Hugging Face Spaces
    - 📊 **Dashboard Frontend** : Streamlit sur Space compagnon
    - 🤖 **Modèle ML** : XGBoost avec preprocessing intégré (source MLflow)
    - 🔬 **Monitoring** : MLflow léger pour production
    """,
    version="1.0.0",
    contact={
        "name": "GetAround Data Science Team",
        "url": "https://github.com/FLebrun67/getaround-project"
    },
    lifespan=lifespan
)

# ✅ Page d'accueil HTML (mise à jour pour HF + modèle info)
@app.get("/", response_class=HTMLResponse)
def root():
    """
    Page d'accueil avec interface HTML moderne et informations du modèle
    """
    # Statut du modèle pour affichage dynamique
    model_status = "✅ Opérationnel" if loaded_model else "❌ Indisponible"
    model_type = type(loaded_model).__name__ if loaded_model else "Aucun"
    status_color = "#27ae60" if loaded_model else "#e74c3c"
    
    # Informations du modèle depuis métadonnées
    model_info = ""
    if model_metadata:
        r2 = model_metadata.get('metrics', {}).get('R2', 'N/A')
        rmse = model_metadata.get('metrics', {}).get('RMSE', 'N/A')
        run_id = model_metadata.get('run_id', 'Unknown')[:8]  # Raccourci
        model_info = f"R² = {r2:.3f}, RMSE = {rmse:.1f}€ (Run: {run_id})"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GetAround API - Prédiction des Prix</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: #333;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .hero {{
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                padding: 40px;
                margin-bottom: 30px;
                backdrop-filter: blur(10px);
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                text-align: center;
            }}
            
            .hero h1 {{
                font-size: 3em;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 20px;
            }}
            
            .hero p {{
                font-size: 1.2em;
                color: #666;
                margin-bottom: 30px;
                line-height: 1.6;
            }}
            
            .status-card {{
                background: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                border-left: 5px solid {status_color};
            }}
            
            .status-title {{
                font-size: 1.3em;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }}
            
            .status-info {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }}
            
            .status-item {{
                background: #f8f9fa;
                padding: 10px 15px;
                border-radius: 8px;
                border-left: 3px solid #3498db;
            }}
            
            .cards-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .card {{
                background: rgba(255, 255, 255, 0.9);
                border-radius: 15px;
                padding: 25px;
                backdrop-filter: blur(10px);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.2);
            }}
            
            .card h3 {{
                color: #2c3e50;
                margin-bottom: 15px;
                font-size: 1.3em;
            }}
            
            .card p {{
                color: #666;
                line-height: 1.6;
                margin-bottom: 20px;
            }}
            
            .btn {{
                display: inline-block;
                padding: 12px 25px;
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-weight: bold;
                transition: all 0.3s ease;
                border: none;
                cursor: pointer;
                margin: 5px;
            }}
            
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(52, 152, 219, 0.3);
            }}
            
            .btn-primary {{ background: linear-gradient(135deg, #e74c3c, #c0392b); }}
            .btn-success {{ background: linear-gradient(135deg, #27ae60, #229954); }}
            .btn-info {{ background: linear-gradient(135deg, #f39c12, #e67e22); }}
            .btn-mlflow {{ background: linear-gradient(135deg, #9b59b6, #8e44ad); }}
            
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                color: white;
            }}
            
            .model-perf {{
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
                color: #666;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Section Hero -->
            <div class="hero">
                <h1>🚗 GetAround API</h1>
                <p>
                    Prédiction intelligente des prix de location de véhicules<br>
                    Alimenté par l'Intelligence Artificielle et le Machine Learning<br>
                    🚀 <strong>Déployé sur Hugging Face Spaces</strong>
                </p>
                <div class="model-perf">{model_info}</div>
                <a href="/docs" class="btn btn-primary">📚 Documentation Interactive</a>
                <a href="/mlflow-stats" class="btn btn-mlflow">🔬 Statistiques MLflow</a>
                <a href="/predict-example" class="btn btn-success">🧪 Voir un Exemple</a>
            </div>
            
            <!-- Statut du Système -->
            <div class="status-card">
                <div class="status-title">📊 État du Système</div>
                <div class="status-info">
                    <div class="status-item">
                        <strong>🤖 Modèle ML:</strong> {model_status}
                    </div>
                    <div class="status-item">
                        <strong>🔧 Type:</strong> {model_type}
                    </div>
                    <div class="status-item">
                        <strong>📡 API:</strong> ✅ En ligne
                    </div>
                    <div class="status-item">
                        <strong>🔬 Source:</strong> {model_source}
                    </div>
                </div>
            </div>
            
            <!-- Cartes de Fonctionnalités -->
            <div class="cards-grid">
                <div class="card">
                    <h3>🎯 Prédiction de Prix</h3>
                    <p>
                        Obtenez une estimation précise du prix de location optimal 
                        basée sur les caractéristiques de votre véhicule.
                    </p>
                    <a href="/docs#/default/predict_predict_post" class="btn">Tester Maintenant</a>
                </div>
                
                <div class="card">
                    <h3>🔬 Monitoring MLflow</h3>
                    <p>
                        Chaque prédiction est automatiquement enregistrée et analysée
                        pour un monitoring de production avancé.
                    </p>
                    <a href="/mlflow-stats" class="btn btn-mlflow">Voir les Stats</a>
                </div>
                
                <div class="card">
                    <h3>📋 Validation Automatique</h3>
                    <p>
                        Interface intelligente qui valide automatiquement vos données 
                        et vous guide en cas d'erreur.
                    </p>
                    <a href="/docs" class="btn">Découvrir</a>
                </div>
                
                <div class="card">
                    <h3>🔍 Monitoring</h3>
                    <p>
                        Surveillez l'état de l'API et du modèle ML en temps réel 
                        avec nos endpoints de monitoring.
                    </p>
                    <a href="/health" class="btn">Voir l'État</a>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>
                    🚗 <strong>GetAround Price Prediction API</strong> - 
                    Développé avec FastAPI, XGBoost, MLflow et ❤️
                </p>
                <p style="margin-top: 10px; opacity: 0.8;">
                    🚀 Hébergé sur <strong>Hugging Face Spaces</strong> avec monitoring <strong>MLflow</strong>
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# ✅ Endpoint health (ajout infos modèle et MLflow)
@app.get("/health")
def health():
    """
    Endpoint de vérification de l'état de l'API
    """
    # Test de l'état MLflow
    mlflow_status = "unknown"
    try:
        current_exp = mlflow.get_experiment_by_name("hf_production_monitoring")
        mlflow_status = "active" if current_exp else "inactive"
    except:  # noqa: E722
        mlflow_status = "error"
    
    return {
        "status": "healthy" if loaded_model and hasattr(loaded_model, 'predict') else "degraded",
        "model_loaded": loaded_model is not None,
        "model_type": type(loaded_model).__name__ if loaded_model else None,
        "model_source": model_source,
        "model_has_predict": hasattr(loaded_model, 'predict') if loaded_model else False,
        "api_version": "1.0.0",
        "deployment": "huggingface_spaces",
        "mlflow_status": mlflow_status,
        "mlflow_dir": mlflow_dir,
        "model_metadata": model_metadata
    }

# ✅ Endpoint model-info (enrichi avec métadonnées)
@app.get("/model-info")
def model_info():
    """
    Informations détaillées sur le modèle chargé
    """
    if loaded_model is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non disponible - vérifiez que le fichier modèle existe"
        )
    
    info = {
        "model_type": type(loaded_model).__name__,
        "model_ready": hasattr(loaded_model, 'predict'),
        "model_source": model_source,
        "metadata": model_metadata,
        "mlflow_enabled": True
    }
    
    # Informations spécifiques au modèle (identiques)
    if hasattr(loaded_model, 'n_features_in_'):
        info["n_features_expected"] = loaded_model.n_features_in_
    
    if hasattr(loaded_model, 'feature_importances_'):
        info["has_feature_importance"] = True
    
    if hasattr(loaded_model, 'get_params'):
        info["model_parameters"] = loaded_model.get_params()
    
    return info

# 🔬 Endpoint MLflow stats (IDENTIQUE à la version complète)
@app.get("/mlflow-stats")
def get_mlflow_stats():
    """
    Statistiques des prédictions depuis MLflow
    Endpoint unique pour monitoring de production
    """
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("hf_production_monitoring")
        
        if not experiment:
            return {
                "status": "no_experiment",
                "message": "Aucune expérience MLflow trouvée",
                "total_predictions": 0
            }
        
        # Récupération des runs (prédictions)
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=1000,
            order_by=["start_time DESC"]
        )
        
        if not runs:
            return {
                "status": "no_runs", 
                "message": "Aucune prédiction enregistrée",
                "total_predictions": 0
            }
        
        # Analyse des prédictions
        predictions = []
        confidence_scores = []
        fuel_types = {}
        brands = {}
        
        for run in runs:
            # Filtrer seulement les runs de prédiction
            if run.data.tags.get("type") == "production_prediction":
                if "predicted_price" in run.data.metrics:
                    predictions.append(run.data.metrics["predicted_price"])
                
                if "confidence_score" in run.data.metrics:
                    confidence_scores.append(run.data.metrics["confidence_score"])
                
                fuel = run.data.tags.get("fuel_type", "unknown")
                fuel_types[fuel] = fuel_types.get(fuel, 0) + 1
                
                brand = run.data.tags.get("brand", "unknown")
                brands[brand] = brands.get(brand, 0) + 1
        
        # Calcul des statistiques
        if predictions:
            stats = {
                "status": "success",
                "total_predictions": len(predictions),
                "price_stats": {
                    "avg_price": round(sum(predictions) / len(predictions), 2),
                    "min_price": round(min(predictions), 2),
                    "max_price": round(max(predictions), 2),
                    "median_price": round(sorted(predictions)[len(predictions)//2], 2)
                },
                "confidence_stats": {
                    "avg_confidence": round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0,
                    "high_confidence_ratio": len([c for c in confidence_scores if c > 0.8]) / len(confidence_scores) if confidence_scores else 0
                },
                "fuel_distribution": fuel_types,
                "brand_distribution": dict(sorted(brands.items(), key=lambda x: x[1], reverse=True)),
                "last_updated": datetime.now().isoformat(),
                "experiment_id": experiment.experiment_id,
                "model_source": model_source,
                "original_model": model_metadata.get('run_id', 'Unknown') if model_metadata else 'Unknown'
            }
        else:
            stats = {
                "status": "no_predictions",
                "message": "Aucune prédiction de type 'production_prediction' trouvée",
                "total_predictions": 0,
                "total_runs": len(runs)
            }
        
        return stats
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erreur lors de la récupération des stats MLflow : {str(e)}",
            "total_predictions": 0
        }

# 🔄 Endpoint de prédiction principal (IDENTIQUE avec ajout logging et timing)
@app.post("/predict", response_model=PricePrediction)
async def predict(features: CarFeatures):
    """
    Prédiction du prix de location journalier avec logging MLflow automatique
    
    **Paramètres:**
    - features: Caractéristiques du véhicule (voir le schéma CarFeatures)
    
    **Retourne:**
    - rental_price: Prix prédit en euros par jour
    - currency: Devise (EUR)
    - period: Période (per_day)
    - status: Statut de la prédiction
    - model_confidence: Niveau de confiance du modèle
    
    **Monitoring:**
    - Chaque prédiction est automatiquement enregistrée dans MLflow
    - Statistiques disponibles sur /mlflow-stats
    """
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Service temporairement indisponible - modèle ML non chargé")

    print("📥 Requête reçue dans /predict")
    start_time = time.time()
    
    try:
        # Préparation des données (IDENTIQUE à ton main2.py)
        input_dict = features.model_dump()
        print("🔍 Données d'entrée :", input_dict)
        
        input_df = pd.DataFrame([input_dict])
        
        # Prédiction avec le modèle
        prediction = loaded_model.predict(input_df)
        predicted_price = float(prediction[0])
        
        # Calcul du temps de traitement
        processing_time = (time.time() - start_time) * 1000  # en ms

        # Logique de validation des prix (IDENTIQUE)
        if predicted_price < 1:
            predicted_price = 30
            confidence = "low"
        elif predicted_price > 1000:
            predicted_price = min(predicted_price, 1000)
            confidence = "medium"
        else:
            confidence = "high"

        final_price = round(predicted_price, 2)

        # 🔬 Logging de la prédiction dans MLflow
        log_prediction_to_mlflow(input_dict, final_price, confidence, processing_time)

        print(f"✅ Prédiction réussie : {final_price}€/jour (confiance: {confidence})")

        return PricePrediction(
            rental_price=final_price,
            currency="EUR",
            period="per_day",
            status="success",
            model_confidence=confidence
        )

    except Exception as e:
        import traceback
        print("❌ Erreur lors de la prédiction :", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction: {str(e)}")

# ✅ Endpoint d'exemple (mis à jour pour HF)
@app.get("/predict-example")
def predict_example():
    """
    Exemple de données pour tester l'endpoint /predict
    """
    return {
        "description": "Exemple de données à envoyer à /predict",
        "method": "POST",
        "url": "/predict",
        "example_data": {
            "model_key": "Renault",
            "mileage": 75000,
            "engine_power": 110,
            "fuel": "diesel",
            "paint_color": "white",
            "car_type": "hatchback",
            "private_parking_available": True,
            "has_gps": True,
            "has_air_conditioning": True,
            "automatic_car": False,
            "has_getaround_connect": True,
            "has_speed_regulator": True,
            "winter_tires": False
        },
        "curl_example": """
curl -X POST "https://ton-username-getaround-api.hf.space/predict" \\
     -H "Content-Type: application/json" \\
     -d '{
         "model_key": "Renault",
         "mileage": 75000,
         "engine_power": 110,
         "fuel": "diesel",
         "paint_color": "white",
         "car_type": "hatchback",
         "private_parking_available": true,
         "has_gps": true,
         "has_air_conditioning": true,
         "automatic_car": false,
         "has_getaround_connect": true,
         "has_speed_regulator": true,
         "winter_tires": false
     }'
        """,
        "model_info": {
            "source": model_source,
            "type": type(loaded_model).__name__ if loaded_model else "Unknown",
            "original_run": model_metadata.get('run_id', 'Unknown') if model_metadata else 'Unknown'
        }
    }

# 🔬 NOUVEAU : Endpoint pour reset/clear des stats MLflow (utile pour demo)
@app.post("/mlflow-reset")
def reset_mlflow_stats():
    """
    Reset des statistiques MLflow (pour démo propre)
    ⚠️ À utiliser avec précaution - efface l'historique des prédictions
    """
    try:
        # Créer une nouvelle expérience avec timestamp
        new_exp_name = f"hf_production_monitoring_{int(time.time())}"
        mlflow.create_experiment(new_exp_name)
        mlflow.set_experiment(new_exp_name)
        
        return {
            "status": "success",
            "message": f"Nouvelle expérience créée : {new_exp_name}",
            "note": "L'ancienne expérience est conservée mais non active"
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Erreur lors du reset : {str(e)}"
        }

# Point d'entrée pour l'exécution (adapté pour HF Spaces)
if __name__ == "__main__":
    import uvicorn
    print("🚀 Démarrage de l'API GetAround sur Hugging Face...")
    print("📁 Recherche du modèle dans le répertoire courant...")
    
    # Vérification des fichiers nécessaires
    required_files = ["trained_model.pkl"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("⚠️ Fichiers manquants :", missing_files)
        print("💡 Assure-toi d'avoir uploadé tous les fichiers depuis hf_deployment/api/")
    
    # Démarrage sur le port standard HF (7860)
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)