import mlflow
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import mlflow.sklearn
from xgboost import XGBRegressor

# Configuration locale MLflow
def configure_mlflow_local():
    mlflow.set_tracking_uri("file:///mlruns")
    mlflow.set_experiment("price_prediction_local")
    print("=== Configuration MLflow (Docker-compatible) ===")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Préparation des données
print("📦 Chargement des données...")
df = pd.read_csv('./data/get_around_pricing_project.csv')

# Nettoyage des données
print("🧹 Nettoyage booléens et valeurs négatives...")
df.loc[df['mileage'] < 0, 'mileage'] = 0
bool_columns = ['private_parking_available', 'has_gps', 'has_air_conditioning', 
                'automatic_car', 'has_getaround_connect', 'has_speed_regulator', 
                'winter_tires']

for col in bool_columns:
    df[col] = df[col].astype(str).map({'True': 1, 'False': 0})


X = df.drop(['rental_price_per_day', 'Unnamed: 0'], axis=1)
y = df['rental_price_per_day']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Colonnes à transformer
numeric_features = ['mileage', 'engine_power', 'private_parking_available', 'has_gps',
                   'has_air_conditioning', 'automatic_car', 'has_getaround_connect', 
                   'has_speed_regulator', 'winter_tires']
categorical_features = ['model_key', 'fuel', 'paint_color', 'car_type']

# Pipeline de prétraitement
def create_pipeline():    
    # Preprocessing numérique
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing catégoriel
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

# RMSE manuel pour éviter l'erreur de compatibilité
def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Entraînement + Logging MLflow
def train_evaluate_model_with_mlflow(model, X_train, X_test, y_train, y_test, model_name):
    print(f"\n=== Entraînement {model_name} ===")
    
    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")
        model.fit(X_train, y_train)

        y_pred_sample = model.predict(X_test)
        signature = infer_signature(X_test, y_pred_sample)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.iloc[:1],
            signature=signature,
            registered_model_name=None  # Optionnel si tu n’utilises pas le registry
            )

        y_pred = model.predict(X_test)
        metrics = {
            "RMSE": compute_rmse(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

        for name, value in metrics.items():
            mlflow.log_metric(name, value)
            print(f"{name}: {value:.2f}")

        return model, run.info.run_id

if __name__ == "__main__":
    configure_mlflow_local()

    model = Pipeline([
        ('preprocessor', create_pipeline()),
        ('regressor', XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.15,
            subsample=0.9,
            random_state=42,
            eval_metric='rmse',
        ))
    ])

    _, run_id = train_evaluate_model_with_mlflow(
        model, X_train, X_test, y_train, y_test, "xgboost_model"
    )
    print(f"Run ID: {run_id}")