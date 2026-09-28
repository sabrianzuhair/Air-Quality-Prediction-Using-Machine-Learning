
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


LINEAR_REGRESSION_MODEL = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\linear_regression_aqi.joblib")
RANDOM_FOREST_MODEL = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\random_forest_aqi.joblib")


LINEAR_REGRESSION_METADATA = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\linear_regression_metadata.json")
RANDOM_FOREST_METADATA = os.path.join(PROJECT_ROOT, "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\RandomForrest_metadata.json")


TEST_DATASET = os.path.join(PROJECT_ROOT, "Test", "C:\\Air Quality Prediction Using Machine Learning\\code\\Test\\dataset_cuaca_aqi.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "ModelComparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
PLOT_LINE = os.path.join(OUTPUT_DIR, "01_actual_vs_predictions_timeline.png")
PLOT_METRICS = os.path.join(OUTPUT_DIR, "02_metrics_comparison.png")
PLOT_SCATTER_LR = os.path.join(OUTPUT_DIR, "03_scatter_lr_vs_actual.png")
PLOT_SCATTER_RF = os.path.join(OUTPUT_DIR, "04_scatter_rf_vs_actual.png")
PLOT_SCATTER_COMPARISON = os.path.join(OUTPUT_DIR, "05_scatter_comparison.png")


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# membuat model machine learning yang sudah dilatih beserta metadata nya
def load_model_and_metadata(model_path: str, metadata_path: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"Loading {model_name}")
    print(f"{'='*60}")
    
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    print(f"✅ Model loaded from: {model_path}")
    
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print(f"✅ Metadata loaded from: {metadata_path}")
    
    
    print(f"   Model type: {metadata.get('model_type', 'Unknown')}")
    print(f"   Version: {metadata.get('model_version', 'Unknown')}")
    print(f"   Training R²: {metadata.get('metrics', {}).get('train_r2', 'N/A'):.4f}")
    print(f"   Test R²: {metadata.get('metrics', {}).get('test_r2', 'N/A'):.4f}")
    
    return model, metadata

#membuat dan membersihkan dataset uji
def load_test_dataset(dataset_path: str):
    
    print(f"\n{'='*60}")
    print("Loading Test Dataset")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Test dataset not found: {dataset_path}")
    
    df = pd.read_csv(dataset_path, encoding="utf-8")
    print(f"✅ Dataset loaded: {len(df)} rows")
    
    
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    
    
    numeric_cols = ["temperature", "humidity", "wind_speed", "pm25", "pm10", "aqi"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    
    required_cols = ["datetime", "city", "temperature", "humidity", "wind_speed", "aqi"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    
    if "aqi" in df.columns:
        df["aqi_actual"] = df["aqi"]
    
    
    initial_count = len(df)
    df = df.dropna(subset=["aqi_actual"])
    dropped_count = initial_count - len(df)
    
    if dropped_count > 0:
        print(f"⚠️  Dropped {dropped_count} rows without actual AQI")
    
    print(f"✅ Final dataset: {len(df)} rows with actual AQI")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    print(f"   Cities: {df['city'].nunique()} unique cities")
    
    return df

#menyesuaikan format fitur agar cocok dengan linear regression
def prepare_features_for_linear_regression(df: pd.DataFrame, metadata: dict):
   
    from sklearn.preprocessing import LabelEncoder
    
    df_prep = df.copy()
    
   
    features_from_metadata = metadata.get("features", [])
    
    if "city_encoded" in features_from_metadata:
        
        cities_from_metadata = metadata.get("cities", [])
        
        
        le = LabelEncoder()
        le.fit(cities_from_metadata)
        
        
        df_prep["city_encoded"] = le.transform(df_prep["city"].astype(str))
        
        
        feature_cols = ["temperature", "humidity", "wind_speed", "pm25", "pm10", "city_encoded"]
    elif "city" in features_from_metadata:
        
        feature_cols = ["temperature", "humidity", "wind_speed", "pm25", "pm10", "city"]
    else:
        
        cities_from_metadata = metadata.get("cities", [])
        le = LabelEncoder()
        le.fit(cities_from_metadata)
        df_prep["city_encoded"] = le.transform(df_prep["city"].astype(str))
        feature_cols = ["temperature", "humidity", "wind_speed", "pm25", "pm10", "city_encoded"]
    
    return df_prep[feature_cols], df_prep

#menyiapkan fitur untuk random forest
def prepare_features_for_random_forest(df: pd.DataFrame):
   
    df_prep = df.copy()
   
    feature_cols = ["temperature", "humidity", "wind_speed", "pm25", "pm10", "city"]
    
    return df_prep[feature_cols], df_prep

#mebuat prediksi aqi dari model
def make_predictions(model, features_df: pd.DataFrame, model_name: str):

    print(f"\nMaking predictions with {model_name}...")
    
    try:
        predictions = model.predict(features_df)
        print(f"✅ Generated {len(predictions)} predictions")
        return predictions
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise


#menghitung metrik performa model
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  R² Score:  {r2:.4f}")
    print(f"  MAE:       {mae:.2f} AQI points")
    print(f"  RMSE:      {rmse:.2f} AQI points")
    
    return metrics

#membuat visualisasi perbandingan hasil prediksi
def create_visualizations(df_results: pd.DataFrame, metrics_lr: dict, metrics_rf: dict):
   
    print(f"\n{'='*60}")
    print("Generating Visualizations")
    print(f"{'='*60}")
    
    
    df_plot = df_results.sort_values("datetime").copy()
    
    print("Creating timeline plot...")
    plt.figure(figsize=(14, 6))
    plt.plot(df_plot["datetime"], df_plot["aqi_actual"], 
             label="Actual AQI", linewidth=2, alpha=0.7, color="black")
    plt.plot(df_plot["datetime"], df_plot["aqi_lr"], 
             label="Linear Regression", linewidth=1.5, alpha=0.7, color="blue")
    plt.plot(df_plot["datetime"], df_plot["aqi_rf"], 
             label="Random Forest", linewidth=1.5, alpha=0.7, color="red")
    plt.xlabel("Date & Time", fontsize=12)
    plt.ylabel("AQI", fontsize=12)
    plt.title("Actual vs Predicted AQI Over Time", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_LINE, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {PLOT_LINE}")
    plt.close()
    
    
    print("Creating metrics comparison bar chart...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = ["Linear\nRegression", "Random\nForest"]
    r2_values = [metrics_lr["r2"], metrics_rf["r2"]]
    mae_values = [metrics_lr["mae"], metrics_rf["mae"]]
    rmse_values = [metrics_lr["rmse"], metrics_rf["rmse"]]
    
  
    axes[0].bar(models, r2_values, color=["#3498db", "#e74c3c"], alpha=0.7)
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].set_title("R² Score Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(r2_values):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")
    
   
    axes[1].bar(models, mae_values, color=["#3498db", "#e74c3c"], alpha=0.7)
    axes[1].set_ylabel("MAE (AQI points)", fontsize=11)
    axes[1].set_title("MAE Comparison", fontsize=12, fontweight="bold")
    for i, v in enumerate(mae_values):
        axes[1].text(i, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    
    axes[2].bar(models, rmse_values, color=["#3498db", "#e74c3c"], alpha=0.7)
    axes[2].set_ylabel("RMSE (AQI points)", fontsize=11)
    axes[2].set_title("RMSE Comparison", fontsize=12, fontweight="bold")
    for i, v in enumerate(rmse_values):
        axes[2].text(i, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(PLOT_METRICS, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {PLOT_METRICS}")
    plt.close()
    
    
    print("Creating Linear Regression scatter plot...")
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results["aqi_actual"], df_results["aqi_lr"], 
                alpha=0.6, s=50, color="blue", edgecolors="black", linewidth=0.5)
    
    
    min_val = min(df_results["aqi_actual"].min(), df_results["aqi_lr"].min())
    max_val = max(df_results["aqi_actual"].max(), df_results["aqi_lr"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("Actual AQI", fontsize=12)
    plt.ylabel("Predicted AQI (Linear Regression)", fontsize=12)
    plt.title(f"Linear Regression: Predicted vs Actual AQI\nR² = {metrics_lr['r2']:.3f}, MAE = {metrics_lr['mae']:.2f}", 
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER_LR, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {PLOT_SCATTER_LR}")
    plt.close()
    
    
    print("Creating Random Forest scatter plot...")
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results["aqi_actual"], df_results["aqi_rf"], 
                alpha=0.6, s=50, color="red", edgecolors="black", linewidth=0.5)
    
    
    min_val = min(df_results["aqi_actual"].min(), df_results["aqi_rf"].min())
    max_val = max(df_results["aqi_actual"].max(), df_results["aqi_rf"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("Actual AQI", fontsize=12)
    plt.ylabel("Predicted AQI (Random Forest)", fontsize=12)
    plt.title(f"Random Forest: Predicted vs Actual AQI\nR² = {metrics_rf['r2']:.3f}, MAE = {metrics_rf['mae']:.2f}", 
              fontsize=13, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER_RF, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {PLOT_SCATTER_RF}")
    plt.close()
    
    
    print("Creating side-by-side scatter comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    
    axes[0].scatter(df_results["aqi_actual"], df_results["aqi_lr"], 
                     alpha=0.6, s=50, color="blue", edgecolors="black", linewidth=0.5)
    min_val = min(df_results["aqi_actual"].min(), df_results["aqi_lr"].min())
    max_val = max(df_results["aqi_actual"].max(), df_results["aqi_lr"].max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 
                 "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual AQI", fontsize=11)
    axes[0].set_ylabel("Predicted AQI", fontsize=11)
    axes[0].set_title(f"Linear Regression\nR² = {metrics_lr['r2']:.3f}, MAE = {metrics_lr['mae']:.2f}", 
                      fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    
    axes[1].scatter(df_results["aqi_actual"], df_results["aqi_rf"], 
                    alpha=0.6, s=50, color="red", edgecolors="black", linewidth=0.5)
    min_val = min(df_results["aqi_actual"].min(), df_results["aqi_rf"].min())
    max_val = max(df_results["aqi_actual"].max(), df_results["aqi_rf"].max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 
                 "r--", linewidth=2, label="Perfect Prediction")
    axes[1].set_xlabel("Actual AQI", fontsize=11)
    axes[1].set_ylabel("Predicted AQI", fontsize=11)
    axes[1].set_title(f"Random Forest\nR² = {metrics_rf['r2']:.3f}, MAE = {metrics_rf['mae']:.2f}", 
                      fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PLOT_SCATTER_COMPARISON, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {PLOT_SCATTER_COMPARISON}")
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {OUTPUT_DIR}")

#menyimpan hasil prediksi ke file csv
def save_results_csv(df_results: pd.DataFrame):
    
    output_cols = ["datetime", "city", "aqi_actual", "aqi_lr", "aqi_rf", "error_lr", "error_rf"]
    
    #
    df_output = df_results[output_cols].copy()
    df_output["datetime"] = df_output["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    df_output.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Results saved to: {RESULTS_CSV}")
    print(f"   Total rows: {len(df_output)}")

#menghasilkan ringkasan akhir dan analisis perbandingan model
def print_final_summary(metrics_lr: dict, metrics_rf: dict, metadata_lr: dict, metadata_rf: dict):
    
    print(f"\n{'='*70}")
    print("FINAL SUMMARY & ANALYSIS")
    print(f"{'='*70}")
    
    
    better_model = "Random Forest" if metrics_rf["r2"] > metrics_lr["r2"] else "Linear Regression"
    r2_diff = abs(metrics_rf["r2"] - metrics_lr["r2"])
    mae_improvement = metrics_lr["mae"] - metrics_rf["mae"]
    rmse_improvement = metrics_lr["rmse"] - metrics_rf["rmse"]
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"   {'Metric':<20} {'Linear Regression':<25} {'Random Forest':<25}")
    print(f"   {'-'*70}")
    print(f"   {'R² Score':<20} {metrics_lr['r2']:<25.4f} {metrics_rf['r2']:<25.4f}")
    print(f"   {'MAE':<20} {metrics_lr['mae']:<25.2f} {metrics_rf['mae']:<25.2f}")
    print(f"   {'RMSE':<20} {metrics_lr['rmse']:<25.2f} {metrics_rf['rmse']:<25.2f}")
    
    print(f"\n🏆 WINNER: {better_model}")
    if better_model == "Random Forest":
        print(f"   • Random Forest achieves {r2_diff:.4f} higher R² score")
        print(f"   • Random Forest reduces MAE by {mae_improvement:.2f} AQI points ({mae_improvement/metrics_lr['mae']*100:.1f}% improvement)")
        print(f"   • Random Forest reduces RMSE by {rmse_improvement:.2f} AQI points ({rmse_improvement/metrics_lr['rmse']*100:.1f}% improvement)")
    else:
        print(f"   • Linear Regression achieves {r2_diff:.4f} higher R² score")
        print(f"   • Linear Regression reduces MAE by {abs(mae_improvement):.2f} AQI points")
        print(f"   • Linear Regression reduces RMSE by {abs(rmse_improvement):.2f} AQI points")
    
    
    print(f"\n🔍 OVERFITTING ANALYSIS:")
    train_r2_lr = metadata_lr.get("metrics", {}).get("train_r2", None)
    test_r2_lr = metadata_lr.get("metrics", {}).get("test_r2", None)
    train_r2_rf = metadata_rf.get("metrics", {}).get("train_r2", None)
    test_r2_rf = metadata_rf.get("metrics", {}).get("test_r2", None)
    
    if train_r2_lr and test_r2_lr:
        gap_lr = train_r2_lr - test_r2_lr
        print(f"   Linear Regression:")
        print(f"     Train R²: {train_r2_lr:.4f}, Test R²: {test_r2_lr:.4f}, Gap: {gap_lr:.4f}")
        if gap_lr > 0.2:
            print(f"     ⚠️  SIGNIFICANT OVERFITTING detected (gap > 0.2)")
        elif gap_lr > 0.1:
            print(f"     ⚠️  Moderate overfitting detected (gap > 0.1)")
        else:
            print(f"     ✅ Good generalization (gap ≤ 0.1)")
    
    if train_r2_rf and test_r2_rf:
        gap_rf = train_r2_rf - test_r2_rf
        print(f"   Random Forest:")
        print(f"     Train R²: {train_r2_rf:.4f}, Test R²: {test_r2_rf:.4f}, Gap: {gap_rf:.4f}")
        if gap_rf > 0.2:
            print(f"     ⚠️  SIGNIFICANT OVERFITTING detected (gap > 0.2)")
        elif gap_rf > 0.1:
            print(f"     ⚠️  Moderate overfitting detected (gap > 0.1)")
        else:
            print(f"     ✅ Good generalization (gap ≤ 0.1)")
    
    
    print(f"\n💡 WHY {better_model.upper()} PERFORMS BETTER:")
    
    if better_model == "Random Forest":
        print(f"   • Random Forest can capture non-linear relationships between features and AQI")
        print(f"   • It handles complex interactions between weather variables and city-specific patterns")
        print(f"   • OneHotEncoder for city provides better representation than LabelEncoder")
        print(f"   • Ensemble method (300 trees) reduces variance and improves generalization")
        print(f"   • Feature importance analysis shows which features matter most")
        
        if metrics_rf["r2"] < 0.5:
            print(f"   • However, R² = {metrics_rf['r2']:.3f} suggests room for improvement")
            print(f"   • Consider: feature engineering, more data, or hyperparameter tuning")
    else:
        print(f"   • Linear Regression provides interpretable coefficients")
        print(f"   • Simpler model may generalize better with limited data")
        print(f"   • Lower computational cost and faster predictions")
        print(f"   • Less prone to overfitting on small datasets")
    
    
    print(f"\n📋 RECOMMENDATIONS:")
    if better_model == "Random Forest":
        print(f"   ✅ Use Random Forest for production deployment")
        print(f"   ✅ Random Forest provides better accuracy and handles non-linear patterns")
        if train_r2_rf and test_r2_rf and (train_r2_rf - test_r2_rf) > 0.15:
            print(f"   ⚠️  Consider further regularization to reduce overfitting")
        print(f"   ✅ Model is ready for production use")
    else:
        print(f"   ✅ Linear Regression may be sufficient for this use case")
        print(f"   ✅ Consider Random Forest if more data becomes available")
        print(f"   ✅ Linear Regression is more interpretable for stakeholders")
    
    print(f"\n{'='*70}\n")

#eksekusi utama
# 1. load model dan metadata
# 2. load dataset uji
# 3. persiapkan fitur untuk masing-masing model
# 4. buat prediksi
# 5. hitung metrik performa
# 6. buat visualisasi
# 7. simpan hasil ke csv
# 8. cetak ringkasan akhir dan analisis perbandingan
def main():
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: Linear Regression vs Random Forest")
    print("Air Quality Index Prediction Evaluation")
    print("="*70)
    
    try:
        
        model_lr, metadata_lr = load_model_and_metadata(
            LINEAR_REGRESSION_MODEL, 
            LINEAR_REGRESSION_METADATA, 
            "Linear Regression"
        )
        
        model_rf, metadata_rf = load_model_and_metadata(
            RANDOM_FOREST_MODEL, 
            RANDOM_FOREST_METADATA, 
            "Random Forest"
        )
        
        
        df_test = load_test_dataset(TEST_DATASET)
        
        print(f"\n{'='*60}")
        print("Preparing Features")
        print(f"{'='*60}")
        
        features_lr, df_prep_lr = prepare_features_for_linear_regression(df_test, metadata_lr)
        print("✅ Features prepared for Linear Regression")
        
        features_rf, df_prep_rf = prepare_features_for_random_forest(df_test)
        print("✅ Features prepared for Random Forest")
        
        
        print(f"\n{'='*60}")
        print("Making Predictions")
        print(f"{'='*60}")
        
        predictions_lr = make_predictions(model_lr, features_lr, "Linear Regression")
        predictions_rf = make_predictions(model_rf, features_rf, "Random Forest")
        
        
        df_results = df_test.copy()
        df_results["aqi_lr"] = predictions_lr
        df_results["aqi_rf"] = predictions_rf
        df_results["error_lr"] = abs(df_results["aqi_actual"] - df_results["aqi_lr"])
        df_results["error_rf"] = abs(df_results["aqi_actual"] - df_results["aqi_rf"])
        
        
        print(f"\n{'='*60}")
        print("Calculating Performance Metrics")
        print(f"{'='*60}")
        
        metrics_lr = calculate_metrics(
            df_results["aqi_actual"].values, 
            df_results["aqi_lr"].values, 
            "Linear Regression"
        )
        
        metrics_rf = calculate_metrics(
            df_results["aqi_actual"].values, 
            df_results["aqi_rf"].values, 
            "Random Forest"
        )
        
        
        create_visualizations(df_results, metrics_lr, metrics_rf)
        
        
        save_results_csv(df_results)
        
        
        print_final_summary(metrics_lr, metrics_rf, metadata_lr, metadata_rf)
        
        print(f"✅ Evaluation completed successfully!")
        print(f"   All outputs saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

