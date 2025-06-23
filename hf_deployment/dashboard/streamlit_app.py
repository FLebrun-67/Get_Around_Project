import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="📊 GetAround Dashboard", layout="wide")
st.title("🚗 Dashboard - Prédiction & Analyse Exploratoire")

API_URL = "https://beltzark-getaround-api.hf.space/predict"

# Sidebar pour les paramètres utilisateur
st.sidebar.header("🔧 Paramètres du véhicule")

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

def get_user_input():
    input_data = {}

    # Paramètres principaux - Catégoriels & numériques
    st.sidebar.subheader("🔹 Paramètres principaux")
    for feature, options in categorical_features.items():
        input_data[feature] = st.sidebar.selectbox(feature, options)

    # Options supplémentaires - Booléens
    with st.sidebar.expander("⚙️ Options supplémentaires (à activer si disponibles)"):
        for feature in boolean_features:
            input_data[feature] = st.checkbox(f"{feature.replace('_', ' ').capitalize()}", value=False)

    return input_data

user_input = get_user_input()

# Onglets Streamlit
tabs = st.tabs(["🎯 Prédiction", "📊 Analyse Pricing", "⏱ Analyse Retards"])

with tabs[0]:
    st.subheader("Prédiction de prix")
    if st.button("Prédire le prix"):
        try:
            response = requests.post(API_URL, json=user_input, timeout=10)
            if response.status_code == 200:
                result = response.json()
                st.success(f"💸 Prix estimé : **{result['rental_price']} € / jour**")
                st.info(f"Niveau de confiance : `{result['model_confidence']}`")
            else:
                st.error(f"Erreur API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur de connexion : {e}")

with tabs[1]:
    st.subheader("Analyse exploratoire - Pricing")
    df_pricing = pd.read_csv("get_around_pricing_project.csv")

    st.plotly_chart(px.histogram(df_pricing, x='mileage', nbins=50, title='Distribution du Kilométrage'))
    st.plotly_chart(px.histogram(df_pricing, x='engine_power', nbins=50, title='Distribution de la Puissance Moteur'))
    st.plotly_chart(px.scatter(df_pricing, x='mileage', y='rental_price_per_day', title='Prix vs Kilométrage'))
    st.plotly_chart(px.scatter(df_pricing, x='engine_power', y='rental_price_per_day', title='Prix vs Puissance Moteur'))

    corr = df_pricing[['mileage', 'engine_power', 'rental_price_per_day']].corr().round(2)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        zmin=-1, zmax=1
    ))
    fig_corr.update_layout(title="Matrice de corrélation")
    st.plotly_chart(fig_corr)

    df_pricing['power_segment'] = pd.cut(df_pricing['engine_power'], bins=[0,100,150,200,float('inf')],
                                         labels=['<100CV', '100-150CV', '150-200CV', '>200CV'])
    st.plotly_chart(px.box(df_pricing, x='power_segment', y='rental_price_per_day', title='Prix par segment de puissance'))

with tabs[2]:
    st.subheader("Analyse exploratoire - Retards")
    df_delay = pd.read_excel("get_around_delay_analysis.xlsx", sheet_name="rentals_data")
    df_ended = df_delay[df_delay["state"] == "ended"].copy()

    st.plotly_chart(px.box(df_ended, x='checkin_type', y='delay_at_checkout_in_minutes', title='Retards par type de check-in'))

    connect_delays = df_ended[df_ended["checkin_type"] == "connect"]["delay_at_checkout_in_minutes"].dropna()
    mobile_delays = df_ended[df_ended["checkin_type"] == "mobile"]["delay_at_checkout_in_minutes"].dropna()
    delay_df = pd.concat([
        pd.DataFrame({'delay': connect_delays, 'type': 'Connect'}),
        pd.DataFrame({'delay': mobile_delays, 'type': 'Mobile'})
    ])
    st.plotly_chart(px.histogram(delay_df, x='delay', color='type', nbins=30, title='Histogramme des retards'))

    retard_pct = df_ended.groupby('checkin_type')['delay_at_checkout_in_minutes'].apply(lambda x: (x > 0).mean() * 100)
    st.plotly_chart(px.bar(x=retard_pct.index, y=retard_pct.values, labels={'x': 'Type de Check-in', 'y': 'Pourcentage de retards'},
                           title='Taux de retard (%)'))