import os
import sys
import math
import json
import argparse
import logging
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
from datetime import datetime

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline


#configurasi
DATASET_PATH = "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\dataset_cuaca_aqi.csv"
MODEL_PATH = "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\linear_regression_aqi.joblib"
METADATA_PATH = "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\linear_regression_metadata.json"

#fungsi utama dari function ini adalah mengatur format dan level logging untuk program
#contohnya menampilkan pesan info, warning, dan error selama eksekusi program
def setup_logging() -> None:
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

#fugsi dari function ini adalah membuat dataset dari file csv dan melakukan preprocessing awal
def load_dataset(csv_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
        logging.info(f"✅ Dataset loaded successfully: {len(df)} rows")
    except FileNotFoundError:
        logging.error(f"❌ Dataset not found: {csv_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Failed to read CSV {csv_path}: {e}")
        sys.exit(1)

    
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    
    numeric_cols = ["temperature", "humidity", "wind_speed", "pressure", "pm25", "pm10", "aqi"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

#fungsi dari function ini adalah melakukan validasi kualitas dataset dan menampilkan statistik penting
def validate_dataset(df: pd.DataFrame) -> None:
    required_cols = ["temperature", "humidity", "wind_speed", "pm25", "pm10", "aqi"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")

    logging.info("\n" + "="*60)
    logging.info("DATASET SUMMARY")
    logging.info("="*60)
    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    
    if "city" in df.columns:
        unique_cities = df["city"].nunique()
        logging.info(f"Number of unique cities: {unique_cities}")
        logging.info("Rows per city:")
        city_counts = df["city"].value_counts().sort_index()
        for city, count in city_counts.items():
            logging.info(f"  {city}: {count} rows ({count/len(df)*100:.1f}%)")

    missing = df[required_cols].isna().sum()
    if missing.any():
        logging.warning("Missing values:")
        for col, count in missing[missing > 0].items():
            logging.warning(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    
  
    logging.info("\nDataset Statistics:")
    stats = df[required_cols].describe()
    logging.info(f"\n{stats}")
    
    
    if (df["humidity"].dropna() > 100).any() or (df["humidity"].dropna() < 0).any():
        logging.warning("⚠️  Humidity contains values outside 0-100 range")
    
    if (df["aqi"].dropna() < 0).any():
        logging.warning("⚠️  AQI contains negative values")
    
    logging.info("="*60 + "\n")

#fungsi utama adalah mengonversi nilai AQI menjadi kategori kualitas udara
def aqi_to_category(aqi_value: float) -> str:
    if pd.isna(aqi_value):
        return "Unknown"
    
    v = float(aqi_value)
    if 0 <= v <= 50:
        return "Good"
    elif 51 <= v <= 100:
        return "Moderate"
    elif 101 <= v <= 150:
        return "Unhealthy for Sensitive Groups"
    elif 151 <= v <= 200:
        return "Unhealthy"
    elif 201 <= v <= 300:
        return "Very Unhealthy"
    elif 301 <= v <= 500:
        return "Hazardous"
    else:
        return "Out of Range"


#menentukan fitur x yang akan digunakan buat training model
def prepare_features(df: pd.DataFrame, include_weather: bool = True) -> Tuple[pd.DataFrame, list]:
    df_processed = df.copy()
    
    
    base_features = ["temperature", "humidity", "wind_speed", "pm25", "pm10"]
    numeric_features = []
    
    for feature in base_features:
        if feature in df_processed.columns:
            
            if df_processed[feature].notna().sum() > 0:
                numeric_features.append(feature)
            else:
                logging.warning(f"⚠️  Feature '{feature}' is completely missing/NaN, excluding from model")
        else:
            logging.warning(f"⚠️  Feature '{feature}' not found in dataset, excluding from model")
    
    
    if "pressure" in df_processed.columns:
        if df_processed["pressure"].notna().sum() > 0:
            numeric_features.append("pressure")
        else:
            logging.warning("⚠️  Feature 'pressure' is completely missing/NaN, excluding from model")
    
    
    if include_weather and "weather" in df_processed.columns:
        
        if df_processed["weather"].notna().sum() > 0:
           
            le = LabelEncoder()
            df_processed["weather_encoded"] = le.fit_transform(df_processed["weather"].astype(str))
            numeric_features.append("weather_encoded")
            logging.info(f"Weather categories encoded: {dict(zip(le.classes_, range(len(le.classes_))))}")
        else:
            logging.warning("⚠️  Feature 'weather' is completely missing/NaN, excluding from model")
    
    
    if "city" in df_processed.columns:
        if df_processed["city"].notna().sum() > 0:
            le_city = LabelEncoder()
            df_processed["city_encoded"] = le_city.fit_transform(df_processed["city"].astype(str))
            numeric_features.append("city_encoded")
            city_mapping = dict(zip(le_city.classes_, range(len(le_city.classes_))))
            logging.info(f"City categories encoded: {city_mapping}")
        else:
            logging.warning("⚠️  Feature 'city' is completely missing/NaN, excluding from model")
    
    if len(numeric_features) == 0:
        raise ValueError("❌ No valid features found for training!")
    
    logging.info(f"✅ Using {len(numeric_features)} features: {', '.join(numeric_features)}")
    
    return df_processed, numeric_features

#membuat pipeline ml berdasarkan linear regression
def build_linear_regression_pipeline(numeric_features: list, scale_features: bool = True) -> Pipeline:
   
    steps = []
    
    
    imputer = SimpleImputer(strategy="median")
    
    
    if scale_features:
        steps.append(("imputer", imputer))
        steps.append(("scaler", StandardScaler()))
    else:
        steps.append(("imputer", imputer))
    
    
    steps.append(("model", LinearRegression()))
    
    pipeline = Pipeline(steps=steps)
    return pipeline

#kegunaanya adalah untuk melatih model dan mengevaluasi program
def train_and_evaluate(
    df: pd.DataFrame,
    numeric_features: list,
    test_size: float = 0.25,
    random_state: int = 42,
    scale_features: bool = True
) -> Tuple[Pipeline, dict]:
    target_column = "aqi"
    used_columns = numeric_features + [target_column]
    
    
    missing_cols = [col for col in used_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"❌ Missing required columns: {missing_cols}")
    
    data = df[used_columns].copy()
    
    
    data = data.dropna(subset=[target_column])
    
    if len(data) < 2:
        raise ValueError(f"❌ Not enough data for training. Need at least 2 rows, got {len(data)}")
    
    X = data[numeric_features]
    y = data[target_column]
    
    
    if len(data) < 5:
        logging.warning(f"⚠️  Very small dataset ({len(data)} rows). Using all data for training, no test split.")
        X_train, X_test = X, X
        y_train, y_test = y, y
    else:
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    logging.info(f"Training set: {len(X_train)} samples")
    logging.info(f"Test set: {len(X_test)} samples")
    
    
    pipeline = build_linear_regression_pipeline(numeric_features, scale_features)
    pipeline.fit(X_train, y_train)
    
   
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    
    train_r2 = r2_score(y_train, y_train_pred)
    
    
    if len(X_test) == len(X_train):
        test_r2 = train_r2
        test_mae = mean_absolute_error(y_train, y_train_pred)
        test_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
        logging.warning("⚠️  Test metrics = Train metrics (using all data for training)")
    else:
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    
    metrics = {
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": len(numeric_features),
        "features": numeric_features
    }
    
    return pipeline, metrics

#melakukan k-fold cross validation
def cross_validate_model(
    df: pd.DataFrame,
    numeric_features: list,
    n_splits: int = 5,
    random_state: int = 42,
    scale_features: bool = True
) -> dict:
    
    target_column = "aqi"
    used_columns = numeric_features + [target_column]
    data = df[used_columns].copy()
    data = data.dropna(subset=[target_column])
    
    X = data[numeric_features]
    y = data[target_column]
    
    
    if len(X) < 5:
        logging.warning(f"⚠️  Dataset too small ({len(X)} rows) for cross-validation. Skipping CV.")
        return {
            "cv_r2_mean": float('nan'),
            "cv_r2_std": float('nan'),
            "cv_mae_mean": float('nan'),
            "cv_mae_std": float('nan'),
            "cv_rmse_mean": float('nan'),
            "cv_rmse_std": float('nan'),
            "n_splits": 0
        }
    
    n_splits = min(n_splits, max(2, len(X) // 3))
    
    pipeline = build_linear_regression_pipeline(numeric_features, scale_features)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    
    r2_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    mae_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_mean_absolute_error")
    rmse_scores = -cross_val_score(pipeline, X, y, cv=cv, scoring="neg_root_mean_squared_error")
    
    cv_metrics = {
        "cv_r2_mean": float(r2_scores.mean()),
        "cv_r2_std": float(r2_scores.std()),
        "cv_mae_mean": float(mae_scores.mean()),
        "cv_mae_std": float(mae_scores.std()),
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
        "n_splits": n_splits
    }
    
    logging.info(f"\nCross-Validation Results ({n_splits}-fold):")
    logging.info(f"  R²:  {cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}")
    logging.info(f"  MAE: {cv_metrics['cv_mae_mean']:.2f} ± {cv_metrics['cv_mae_std']:.2f}")
    logging.info(f"  RMSE: {cv_metrics['cv_rmse_mean']:.2f} ± {cv_metrics['cv_rmse_std']:.2f}")
    
    return cv_metrics

#menampilkan persamaan matematis dari model machine learning linear regression
def print_model_equation(pipeline: Pipeline, feature_names: list) -> None:
    
    model = pipeline.named_steps["model"]
    coefficients = model.coef_
    intercept = model.intercept_
    
    print("\n" + "="*60)
    print("LINEAR REGRESSION EQUATION")
    print("="*60)
    print(f"AQI = {intercept:.4f}")
    for coef, feature in zip(coefficients, feature_names):
        sign = "+" if coef >= 0 else "-"
        print(f"      {sign} {abs(coef):.4f} × {feature}")
    print("="*60 + "\n")

#fungsi utamanya adalah memprediksi aqi dan menampilkan dilayar
def predict_and_display(pipeline: Pipeline, df: pd.DataFrame, numeric_features: list, n_samples: int = 10) -> pd.DataFrame:
    
    data = df[numeric_features + ["datetime"]].copy()
    data = data.dropna(subset=numeric_features)
    
    if len(data) == 0:
        logging.warning("⚠️  No valid data for prediction")
        return pd.DataFrame()
    
    
    predictions = pipeline.predict(data[numeric_features])
    data["aqi_predicted"] = predictions
    data["category"] = data["aqi_predicted"].apply(aqi_to_category)
    
    
    if "aqi" in df.columns:
        data["aqi_actual"] = df.loc[data.index, "aqi"]
        data["error"] = abs(data["aqi_actual"] - data["aqi_predicted"])
    
    
    print("\n" + "="*60)
    print("PREDICTION PREVIEW")
    print("="*60)
    display_cols = ["datetime", "aqi_predicted", "category"]
    if "aqi_actual" in data.columns:
        display_cols.insert(-1, "aqi_actual")
        display_cols.append("error")
    
    print(data[display_cols].head(n_samples).to_string(index=False))
    
    
    print("\n" + "="*60)
    print("PREDICTED AQI CATEGORY DISTRIBUTION")
    print("="*60)
    print(data["category"].value_counts().to_string())
    print("="*60 + "\n")
    
    return data

#visualisasi hasil training model machine learning
def visualize_training_results(df: pd.DataFrame, pipeline: Pipeline, numeric_features: list) -> None:
   
    try:
        
        valid_df = df.dropna(subset=numeric_features + ["aqi"]).copy()
        if len(valid_df) == 0:
            logging.warning("No valid rows available for visualization.")
            return

        X = valid_df[numeric_features]
        y_true = valid_df["aqi"]
        y_pred = pipeline.predict(X)

        
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.7)
        min_val = np.nanmin([y_true.min(), y_pred.min()])
        max_val = np.nanmax([y_true.max(), y_pred.max()])
        plt.plot([min_val, max_val], [min_val, max_val], ls="--", color="red", label="Perfect prediction")
        plt.xlabel("Actual AQI")
        plt.ylabel("Predicted AQI")
        plt.title("Actual vs Predicted AQI")
        plt.legend()
        plt.tight_layout()

        
        errors = y_true - y_pred
        plt.figure(figsize=(6, 5))
        sns.histplot(errors, kde=True, bins=20, color="steelblue")
        plt.xlabel("Prediction Error (Actual - Predicted)")
        plt.title("Prediction Error Distribution")
        plt.tight_layout()

        model = pipeline.named_steps.get("model")
        if model is not None and hasattr(model, "coef_"):
            coefs = model.coef_
            plt.figure(figsize=(7, max(4, len(numeric_features) * 0.5)))
            sns.barplot(x=coefs, y=numeric_features, orient="h", palette="viridis")
            plt.xlabel("Coefficient")
            plt.ylabel("Feature")
            plt.title("Feature Importance (Linear Regression Coefficients)")
            plt.tight_layout()
        else:
            logging.warning("⚠️  Cannot plot feature importance: model coefficients not available.")

        plt.show()

    except Exception as e:
        logging.error(f"Visualization error: {e}")

#menyimpan model ke file .joblib
#file model akan tersimpan dalam disk
def save_model(pipeline: Pipeline, path: str) -> None:
   
    joblib.dump(pipeline, path)
    logging.info(f"Model saved to: {path}")


#menyimpan metadata training ke file JSON
def save_metadata(metrics: dict, cv_metrics: dict, df: pd.DataFrame, model_path: str, feature_names: list) -> None:
    
    if "city" in df.columns and df["city"].notna().any():
        unique_cities = df["city"].dropna().unique().tolist()
        unique_cities = sorted([str(c) for c in unique_cities])
        num_cities = len(unique_cities)
        dataset_scope = "multi_city" if num_cities > 1 else "single_city"
    else:
        unique_cities = []
        num_cities = 0
        dataset_scope = "single_city"
    
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_type": "Linear Regression",
        "model_version": "v2-multicity",
        "model_path": model_path,
        "dataset_path": DATASET_PATH,
        "dataset_rows": int(len(df)),
        "dataset_scope": dataset_scope,
        "cities": unique_cities,
        "num_cities": num_cities,
        "dataset_date_range": {
            "start": str(df["datetime"].min()),
            "end": str(df["datetime"].max())
        },
        "features": feature_names,
        "metrics": metrics,
        "cross_validation": cv_metrics
    }
    
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Metadata saved to: {METADATA_PATH}")


#Menampilkan ringkasan hasil training model ke layar
def print_summary(metrics: dict, cv_metrics: dict) -> None:
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: Linear Regression")
    print(f"Training samples: {metrics['n_train']}")
    print(f"Test samples: {metrics['n_test']}")
    print(f"Features: {', '.join(metrics['features'])}")
    print("\nTest Set Performance:")
    print(f"  R² Score:  {metrics['test_r2']:.4f}")
    print(f"  MAE:       {metrics['test_mae']:.2f} AQI points")
    print(f"  RMSE:      {metrics['test_rmse']:.2f} AQI points")
    
    
    if metrics['test_r2'] > 0.99:
        logging.warning("⚠️  WARNING: Test R² > 0.99 detected!")
        logging.warning("   Possible issues:")
        logging.warning("   - Duplicated rows in dataset")
        logging.warning("   - Data leakage (target information in features)")
        logging.warning("   - Very small dataset (insufficient generalization)")
    
    print(f"\nCross-Validation Performance ({cv_metrics['n_splits']}-fold):")
    print(f"  R² Score:  {cv_metrics['cv_r2_mean']:.4f} ± {cv_metrics['cv_r2_std']:.4f}")
    print(f"  MAE:       {cv_metrics['cv_mae_mean']:.2f} ± {cv_metrics['cv_mae_std']:.2f}")
    print(f"  RMSE:      {cv_metrics['cv_rmse_mean']:.2f} ± {cv_metrics['cv_rmse_std']:.2f}")
    print("="*60 + "\n")


#Membuat command-line interface (CLI) untuk program
def build_cli() -> argparse.ArgumentParser:
    
    parser = argparse.ArgumentParser(
        description="Train Linear Regression model for Air Quality Index prediction"
    )
    parser.add_argument(
        "--data",
        default=DATASET_PATH,
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--model",
        default=MODEL_PATH,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test set proportion (default: 0.25)"
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable feature scaling"
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Exclude weather categorical feature"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    return parser

#fungsi utama dari program ini
#1. alur dari program iniadalah :
#2. validasi dataset
#3. persiapan fitur
#4. training dan evaluasi model
#5. Evaluasi cross validation
#6. menampilkan ringkasan hasil training
#7. save model dan metadata

def main() -> None:
    
    setup_logging()
    parser = build_cli()
    args = parser.parse_args()
    
    
    if not os.path.exists(args.data):
        logging.error(f"❌ Dataset not found: {args.data}")
        sys.exit(1)
    
    df = load_dataset(args.data)
    validate_dataset(df)
    
    
    df, numeric_features = prepare_features(df, include_weather=not args.no_weather)
    
    
    logging.info("Training Linear Regression model...")
    pipeline, metrics = train_and_evaluate(
        df,
        numeric_features,
        test_size=args.test_size,
        scale_features=not args.no_scale
    )
    
  
    cv_metrics = cross_validate_model(
        df,
        numeric_features,
        n_splits=args.cv_folds,
        scale_features=not args.no_scale
    )
    
    
    print_summary(metrics, cv_metrics)
    print_model_equation(pipeline, numeric_features)
    
    
    predictions_df = predict_and_display(pipeline, df, numeric_features)

    
    visualize_training_results(df, pipeline, numeric_features)
    
    
    save_model(pipeline, args.model)
    save_metadata(metrics, cv_metrics, df, args.model, numeric_features)
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
