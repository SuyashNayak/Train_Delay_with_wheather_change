import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. Load the Kaggle Dataset
print("Loading data...")
df = pd.read_csv('ir_train.csv')

sample_df = df[['fog_risk_score', 'season_severity_score', 'is_monsoon_season', 'zone_congestion_index', 'seat_utilisation_pct', 'is_delayed']].head(50)
sample_df.to_json('dataset_sample.json', orient='records')
print("Sample dataset exported for frontend view!")

# 2. Select Features (Weather + Operations)
features = [
    'fog_risk_score', 
    'season_severity_score', 
    'is_monsoon_season',
    'zone_congestion_index',
    'late_incoming_rake',
    'seat_utilisation_pct'
]

# Drop missing rows for these specific columns to keep the baseline clean
df = df.dropna(subset=features + ['is_delayed'])

X = df[features]
y = df['is_delayed']

# 3. Train/Test Split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train the Model
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='auc',
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate Performance
y_pred_proba = model.predict_proba(X_test)[:, 1]
print(f"Validation ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

# 6. Save the Model for FastAPI
model.save_model("delay_model.json")
print("Model successfully saved to delay_model.json")

# 7. Generate SHAP Explainability (Crucial for proving weather impact)
print("Generating SHAP summary plot...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("shap_weather_impact.png")
print("SHAP plot saved as shap_weather_impact.png")