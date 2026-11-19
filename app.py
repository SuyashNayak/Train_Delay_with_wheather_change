from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import sqlite3
import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Railway Delay Predictor API")

# Allow the frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all domains/file paths
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = xgb.XGBClassifier()
model.load_model("delay_model.json")

# Initialize SQLite Database for MLOps tracking
DB_NAME = "predictions_log.db"
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            fog_risk_score REAL,
            season_severity_score REAL,
            is_monsoon_season INTEGER,
            zone_congestion_index REAL,
            late_incoming_rake INTEGER,
            seat_utilisation_pct REAL,
            delay_probability REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# Define API input data structure
class TrainData(BaseModel):
    fog_risk_score: float
    season_severity_score: float
    is_monsoon_season: int
    zone_congestion_index: float
    late_incoming_rake: int
    seat_utilisation_pct: float

@app.post("/predict")
async def predict_delay(data: TrainData):
    try:
        input_df = pd.DataFrame([data.dict()])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # Log the payload and result
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO logs (
                timestamp, fog_risk_score, season_severity_score, is_monsoon_season, 
                zone_congestion_index, late_incoming_rake, seat_utilisation_pct, delay_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(), data.fog_risk_score, data.season_severity_score,
            data.is_monsoon_season, data.zone_congestion_index, data.late_incoming_rake, 
            data.seat_utilisation_pct, probability
        ))
        conn.commit()
        conn.close()

        return {"delay_probability": round(probability, 4)}

    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from typing import List, Optional

@app.get("/dataset-paginated")
async def get_dataset_page(page: int = 1, limit: int = 50, columns: Optional[str] = None):
    df = pd.read_csv('ir_train.csv')
    
    if columns:
        col_list = [col.strip() for col in columns.split(",")]
        valid_cols = [col for col in col_list if col in df.columns]
        if valid_cols:
            df = df[valid_cols]

    total_rows = len(df)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    
    subset_df = df.iloc[start_idx:end_idx]
    subset_df = subset_df.where(pd.notnull(subset_df), None)
    
    return {
        "total_rows": total_rows,
        "page": page,
        "limit": limit,
        "total_pages": (total_rows // limit) + 1,
        "columns": list(df.columns),  # <--- Ensures column names are sent explicitly
        "data": subset_df.to_dict(orient="records")
    }


from fastapi.responses import FileResponse

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")