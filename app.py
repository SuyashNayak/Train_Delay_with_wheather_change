import datetime
import json
import os
import sqlite3
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from pydantic import BaseModel
import xgboost as xgb

app = FastAPI(title="Railway Delay Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = xgb.XGBClassifier()
model.load_model("delay_model.json")

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
        input_df = pd.DataFrame([data.model_dump()])
        probability = float(model.predict_proba(input_df)[0][1])

        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO logs (
                timestamp, fog_risk_score, season_severity_score, is_monsoon_season,
                zone_congestion_index, late_incoming_rake, seat_utilisation_pct, delay_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.datetime.now().isoformat(),
                data.fog_risk_score,
                data.season_severity_score,
                data.is_monsoon_season,
                data.zone_congestion_index,
                data.late_incoming_rake,
                data.seat_utilisation_pct,
                probability,
            ),
        )
        conn.commit()
        conn.close()

        return {"delay_probability": round(probability, 4)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def load_dataset_df() -> pd.DataFrame:
    if os.path.exists("ir_train.csv"):
        return pd.read_csv("ir_train.csv")
    if os.path.exists("dataset_sample.json"):
        with open("dataset_sample.json", "r", encoding="utf-8") as f:
            records = json.load(f)
        return pd.DataFrame(records)
    return pd.DataFrame()


@app.get("/dataset-paginated")
async def get_dataset_page(page: int = 1, limit: int = 50, columns: Optional[str] = None):
    df = load_dataset_df()
    if df.empty:
        return {
            "total_rows": 0,
            "page": page,
            "limit": limit,
            "total_pages": 1,
            "columns": [],
            "data": [],
        }

    if columns:
        col_list = [col.strip() for col in columns.split(",")]
        valid_cols = [col for col in col_list if col in df.columns]
        if valid_cols:
            df = df[valid_cols]

    total_rows = len(df)
    start_idx = max(0, (page - 1) * limit)
    end_idx = start_idx + limit

    subset_df = df.iloc[start_idx:end_idx]
    subset_df = subset_df.where(pd.notnull(subset_df), None)

    return {
        "total_rows": total_rows,
        "page": page,
        "limit": limit,
        "total_pages": max(1, (total_rows + limit - 1) // limit),
        "columns": list(df.columns),
        "data": subset_df.to_dict(orient="records"),
    }


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")