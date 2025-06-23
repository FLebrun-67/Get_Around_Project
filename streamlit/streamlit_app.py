# streamlit_app.py - Version Hugging Face adaptée de app.py
# 🚀 À placer dans hf_deployment/dashboard/

import streamlit as st
import pandas as pd
import requests
#import plotly.express as px
#import plotly.graph_objects as go
#from datetime import datetime
#import time

# -------------------
# Config Streamlit - ✅ GARDÉE IDENTIQUE
# -------------------
st.set_page_config(page_title="📊 GetAround Dashboard", layout="wide")
st.title("🚗 Dashboard - Prédiction de Prix GetAround")

# 🔄 MODIFIÉ : URL de l'API vers ton Space HF
# ⚠️ IMPORTANT : Remplace "ton-username" par ton vrai nom d'utilisateur HF
API_URL = "http://api:8000/predict"

# 🔄 NOUVEAU : URLs pour les autres endpoints
API_BASE = API_URL.replace("/predict", "")
HEALTH_URL = f"{API_BASE}/health"
MLFLOW_STATS_URL = f"{API_BASE}/mlflow-stats"
MODEL_INFO_URL = f"{API_BASE}/model-info"

# Affichage des infos de connexion
st.sidebar.info("🔗 **API Endpoint:** " + API_URL)
st.sidebar.info("📝 **Documentation:** [Voir l'API](" + f"{API_BASE}/docs" + ")")

# -------------------
# Features définies - ✅ GARDÉES IDENTIQUES
# -------------------
numeric_features = {
    'mileage': (0, 500_000, 5000),
    'engine_power': (20, 300, 5)
}

boolean_features = [
    'private_parking_available', 'has_gps', 'has_air_conditioning',
    'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 'winter_tires'
]

categorical_features = {
    'model_key': ['Renault', 'Peugeot', 'BMW', 'Audi', 'Ford', 'Volkswagen', 'Mercedes'],
    'fuel': ['diesel', 'petrol', 'hybrid_petrol', 'electro'],
    'paint_color': ['black', 'grey', 'white', 'red', 'silver', 'blue', 'orange'],
    'car_type': ['sedan', 'hatchback', 'estate', 'convertible', 'coupe', 'subcompact']
}

# -------------------
# Sidebar - Paramètres véhicule - ✅ GARDÉE IDENTIQUE
# -------------------
st.sidebar.header("🔧 Paramètres du véhicule")

def get_user_input():
    """
    ✅ FONCTION IDENTIQUE à ton app.py original
    Collecte les paramètres depuis l'interface Streamlit
    """
    input_data = {}

    # Paramètres numériques
    for feature, (min_val, max_val, step) in numeric_features.items():
        input_data[feature] = st.sidebar.slider(feature, min_val, max_val, step=step)

    # Paramètres booléens
    for feature in boolean_features:
        input_data[feature] = st.sidebar.selectbox(f"{feature}", [True, False])

    # Paramètres catégoriels
    for feature, options in categorical_features.items():
        input_data[feature] = st.sidebar.selectbox(feature, options)

    return input_data

user_input = get_user_input()

# 🔄 NOUVEAU : Section de statut de l'API
with st.sidebar:
    st.markdown("---")
    st.subheader("🔗 État de l'API")
    
    if st.button("🏥 Tester la Connexion"):
        try:
            response = requests.get(HEALTH_URL, timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("model_loaded"):
                    st.success("✅ API accessible et modèle chargé")
                    st.json(health_data)
                else:
                    st.warning("⚠️ API accessible mais modèle non chargé")
            else:
                st.error(f"❌ Erreur API : {response.status_code}")
        except requests.exceptions.RequestException:
            st.error("❌ Impossible de contacter l'API")
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

# -------------------
# Section Prédiction - 🔄 MODIFIÉE pour HF
# -------------------
st.subheader("🎯 Prédiction de prix de location")

# 🔄 NOUVEAU : Affichage des données d'entrée
with st.expander("🔍 Voir les données d'entrée"):
    st.json(user_input)

if st.button("Prédire le prix", type="primary"):
    try:
        # 🔄 MODIFIÉ : Appel vers l'API Hugging Face avec gestion d'erreurs améliorée
        with st.spinner("🔄 Connexion à l'API GetAround..."):
            response = requests.post(API_URL, json=user_input, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # ✅ GARDÉ IDENTIQUE : Affichage du résultat principal
            st.success(f"💸 Prix estimé : **{result['rental_price']} € / jour**")
            st.info(f"Niveau de confiance : `{result['model_confidence']}`")
            
            # 🔄 NOUVEAU : Affichage enrichi des résultats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prix journalier", f"{result['rental_price']}€")
            with col2:
                st.metric("Devise", result['currency'])
            with col3:
                confidence_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                confidence_icon = confidence_color.get(result['model_confidence'], "⚪")
                st.metric("Confiance", f"{confidence_icon} {result['model_confidence']}")
            
            # 🔄 NOUVEAU : Estimation mensuelle/annuelle
            monthly_price = result['rental_price'] * 30
            yearly_price = result['rental_price'] * 365
            
            st.markdown("### 📊 Estimations de revenus")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"📅 **Revenus mensuels estimés** : {monthly_price:.0f}€")
            with col2:
                st.info(f"📆 **Revenus annuels estimés** : {yearly_price:.0f}€")
            
            # 🔄 NOUVEAU : Détails de la réponse
            with st.expander("📊 Détails de la prédiction"):
                st.json(result)
                
        elif response.status_code == 503:
            st.error("🚫 Service temporairement indisponible - Le modèle ML n'est pas chargé")
            st.info("💡 Vérifiez l'état de l'API dans la sidebar")
        elif response.status_code == 422:
            st.error("❌ Erreur de validation des données - Vérifiez vos paramètres")
            with st.expander("Détails de l'erreur"):
                st.code(response.text)
        else:
            st.error(f"❌ Erreur {response.status_code}")
            st.code(response.text)
            
    except requests.exceptions.Timeout:
        st.error("⏰ Timeout - L'API met trop de temps à répondre")
        st.info("💡 L'API Hugging Face peut être en cours de démarrage (cold start)")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Erreur de connexion - Vérifiez que l'API est accessible")
        st.error(f"URL testée : {API_URL}")
        st.info("💡 Vérifiez que le Space API est bien démarré")
    except Exception as e:
        st.error(f"❌ Erreur lors de l'appel API : {e}")

# -------------------
# Section exemples de données - 🔄 NOUVEAU pour faciliter les tests
# -------------------
st.markdown("---")
st.subheader("💡 Exemples de Véhicules")

# Exemples prédéfinis pour faciliter les tests
examples = {
    "🚗 Renault Clio Économique": {
        "model_key": "Renault",
        "mileage": 80000,
        "engine_power": 90,
        "fuel": "diesel",
        "paint_color": "white",
        "car_type": "hatchback",
        "private_parking_available": True,
        "has_gps": False,
        "has_air_conditioning": True,
        "automatic_car": False,
        "has_getaround_connect": False,
        "has_speed_regulator": False,
        "winter_tires": False
    },
    "🏎️ BMW Série 3 Premium": {
        "model_key": "BMW",
        "mileage": 45000,
        "engine_power": 180,
        "fuel": "petrol",
        "paint_color": "black",
        "car_type": "sedan",
        "private_parking_available": True,
        "has_gps": True,
        "has_air_conditioning": True,
        "automatic_car": True,
        "has_getaround_connect": True,
        "has_speed_regulator": True,
        "winter_tires": True
    },
    "🌱 Véhicule Électrique": {
        "model_key": "Renault",
        "mileage": 25000,
        "engine_power": 110,
        "fuel": "electro",
        "paint_color": "blue",
        "car_type": "hatchback",
        "private_parking_available": True,
        "has_gps": True,
        "has_air_conditioning": True,
        "automatic_car": True,
        "has_getaround_connect": True,
        "has_speed_regulator": True,
        "winter_tires": False
    },
    "🚙 Ford SUV Familial": {
        "model_key": "Ford",
        "mileage": 60000,
        "engine_power": 150,
        "fuel": "petrol",
        "paint_color": "silver",
        "car_type": "estate",
        "private_parking_available": True,
        "has_gps": True,
        "has_air_conditioning": True,
        "automatic_car": False,
        "has_getaround_connect": True,
        "has_speed_regulator": True,
        "winter_tires": True
    }
}

# Interface pour les exemples
col1, col2 = st.columns([2, 1])
with col1:
    selected_example = st.selectbox("Choisir un exemple à tester :", list(examples.keys()))
with col2:
    test_example = st.button(f"🧪 Tester l'Exemple", type="secondary")

if test_example:
    try:
        example_data = examples[selected_example]
        
        with st.spinner("🔄 Test en cours..."):
            response = requests.post(API_URL, json=example_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            
            # Affichage du résultat avec style
            st.success(f"✅ **{selected_example}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💸 Prix estimé", f"{result['rental_price']} €/jour")
            with col2:
                confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.metric("🎯 Confiance", f"{confidence_emoji.get(result['model_confidence'], '⚪')} {result['model_confidence']}")
            with col3:
                monthly = result['rental_price'] * 30
                st.metric("📅 Revenus/mois", f"{monthly:.0f}€")
            
            # Détails de l'exemple
            with st.expander("🔍 Voir les caractéristiques testées"):
                st.json(example_data)
                
        else:
            st.error(f"❌ Erreur lors du test : {response.status_code}")
            st.code(response.text)
            
    except Exception as e:
        st.error(f"❌ Erreur : {e}")

# -------------------
# Section informations sur le modèle - 🔄 MODIFIÉE (MLflow → API stats)
# -------------------
st.markdown("---")
st.subheader("📈 Informations sur le Modèle & Statistiques")

# Onglets pour organiser les informations
tab1, tab2, tab3 = st.tabs(["🤖 Modèle", "📊 Statistiques MLflow", "📈 Graphiques"])

with tab1:
    st.markdown("### 🤖 Informations du Modèle")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Récupérer les infos du modèle"):
            try:
                response = requests.get(MODEL_INFO_URL, timeout=10)
                if response.status_code == 200:
                    model_data = response.json()
                    st.success("✅ Informations du modèle récupérées")
                    
                    # Affichage organisé des informations
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.metric("Type de modèle", model_data.get("model_type", "N/A"))
                        st.metric("Modèle prêt", "✅ Oui" if model_data.get("model_ready") else "❌ Non")
                    
                    with col_b:
                        if "n_features_expected" in model_data:
                            st.metric("Nombre de features", model_data["n_features_expected"])
                        st.metric("Source", model_data.get("model_source", "N/A"))
                        
                    # Métadonnées complètes
                    if model_data.get("metadata"):
                        metadata = model_data["metadata"]
                        if metadata.get("metrics"):
                            st.markdown("#### 📈 Performances du modèle")
                            metrics = metadata["metrics"]
                            col_x, col_y, col_z = st.columns(3)
                            with col_x:
                                st.metric("R²", f"{metrics.get('R2', 0):.4f}")
                            with col_y:
                                st.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}€")
                            with col_z:
                                st.metric("MAE", f"{metrics.get('MAE', 0):.2f}€")
                    
                    # Détails techniques
                    with st.expander("🔍 Détails techniques complets"):
                        st.json(model_data)
                else:
                    st.error(f"❌ Impossible de récupérer les infos : {response.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur : {e}")
    
    with col2:
        st.info("""
        🚀 **Architecture déployée sur Hugging Face Spaces**
        
        Cette version utilise un modèle pré-entrainé exporté depuis MLflow.
        
        **Modèle actuel :**
        - Algorithme : XGBoost Regressor
        - Pipeline : Preprocessing + Modèle intégré
        - Source : MLflow local
        - Déploiement : Hugging Face Spaces
        
        **Pour l'historique complet des expériences :**
        Consultez la version locale avec Docker Compose et MLflow UI.
        """)

with tab2:
    st.markdown("### 📊 Statistiques de Production (MLflow)")
    
    if st.button("📈 Récupérer les statistiques MLflow"):
        try:
            response = requests.get(MLFLOW_STATS_URL, timeout=15)
            if response.status_code == 200:
                stats = response.json()
                
                if stats.get("status") == "success":
                    st.success(f"✅ {stats['total_predictions']} prédictions analysées")
                    
                    # Statistiques de prix
                    price_stats = stats.get("price_stats", {})
                    if price_stats:
                        st.markdown("#### 💰 Statistiques des prix prédits")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Prix moyen", f"{price_stats.get('avg_price', 0):.2f}€")
                        with col2:
                            st.metric("Prix min", f"{price_stats.get('min_price', 0):.2f}€")
                        with col3:
                            st.metric("Prix max", f"{price_stats.get('max_price', 0):.2f}€")
                        with col4:
                            st.metric("Prix médian", f"{price_stats.get('median_price', 0):.2f}€")
                    
                    # Distribution par carburant
                    fuel_dist = stats.get("fuel_distribution", {})
                    if fuel_dist:
                        st.markdown("#### ⛽ Distribution par type de carburant")
                        fuel_df = pd.DataFrame(list(fuel_dist.items()), columns=["Carburant", "Nombre"])
                        st.bar_chart(fuel_df.set_index("Carburant"))
                    
                    # Distribution par marque
                    brand_dist = stats.get("brand_distribution", {})
                    if brand_dist:
                        st.markdown("#### 🚗 Top marques testées")
                        brand_df = pd.DataFrame(list(brand_dist.items()), columns=["Marque", "Nombre"])
                        st.bar_chart(brand_df.set_index("Marque").head(10))
                    
                    # Statistiques de confiance
                    conf_stats = stats.get("confidence_stats", {})
                    if conf_stats:
                        st.markdown("#### 🎯 Statistiques de confiance")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confiance moyenne", f"{conf_stats.get('avg_confidence', 0):.2f}")
                        with col2:
                            ratio = conf_stats.get('high_confidence_ratio', 0)
                            st.metric("% haute confiance", f"{ratio:.1%}")
                    
                    # Détails complets
                    with st.expander("📊 Données complètes"):
                        st.json(stats)
                        
                elif stats.get("status") == "no_predictions":
                    st.info("ℹ️ Aucune prédiction enregistrée pour le moment")
                    st.info("💡 Faites quelques prédictions pour voir les statistiques !")
                else:
                    st.warning(f"⚠️ {stats.get('message', 'Données non disponibles')}")
                    
            else:
                st.error(f"❌ Erreur lors de la récupération : {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Erreur : {e}")

with tab3:
    st.markdown("### 📈 Visualisations")
    st.info("""
    🚧 **Section en développement**
    
    Les graphiques apparaîtront ici une fois que des prédictions auront été faites.
    
    **Graphiques prévus :**
    - Distribution des prix par marque
    - Évolution des prédictions dans le temps  
    - Heatmap des caractéristiques populaires
    - Comparaison par type de carburant
    """)

# -------------------
# Footer informatif - 🔄 NOUVEAU
# -------------------
st.markdown("---")
st.markdown("""
### 🚀 À propos de cette démonstration

**Architecture Hybride :**
- 🎯 **API Backend** : FastAPI déployé sur Hugging Face Spaces
- 📊 **Dashboard Frontend** : Streamlit déployé sur Hugging Face Spaces  
- 🤖 **Modèle ML** : XGBoost avec preprocessing intégré (exporté depuis MLflow)
- 📡 **Communication** : Appels HTTP REST entre les deux Spaces
- 🔬 **Monitoring** : MLflow léger pour tracking de production

**Version locale complète disponible avec :**
- 🔬 MLflow UI pour le tracking des expériences  
- 🐳 Docker Compose pour l'orchestration
- 📈 Historique complet des modèles et métriques

**Performance du modèle :**
- R² : ~0.75 (explique 75% de la variance des prix)
- RMSE : ~16€ (erreur moyenne de ±16€)
- MAE : ~10€ (erreur absolue moyenne de ±10€)

**Code source :** [GitHub Repository](https://github.com/FLebrun67/getaround-project)
""")

# 🔄 NOUVEAU : Informations de debug en mode développement
if st.sidebar.checkbox("🔧 Mode Debug"):
    st.sidebar.markdown("### 🛠️ Informations Debug")
    st.sidebar.code(f"API_URL = {API_URL}")
    st.sidebar.code(f"User Input Keys: {list(user_input.keys())}")
    
    # Test rapide des endpoints
    st.sidebar.markdown("#### 🔍 Tests rapides")
    endpoints = {
        "Health": "/health",
        "Model Info": "/model-info", 
        "MLflow Stats": "/mlflow-stats",
        "Example": "/predict-example",
        "Docs": "/docs"
    }
    
    for name, endpoint in endpoints.items():
        test_url = API_BASE + endpoint
        if st.sidebar.button(f"Test {name}"):
            try:
                resp = requests.get(test_url, timeout=5)
                st.sidebar.success(f"✅ {name}: {resp.status_code}")
                if resp.status_code == 200:
                    with st.sidebar.expander(f"Réponse {name}"):
                        st.json(resp.json())
            except Exception as e:
                st.sidebar.error(f"❌ {name}: {str(e)[:50]}...")

