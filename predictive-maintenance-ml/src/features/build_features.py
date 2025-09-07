"""Build rolling-window features and forward-looking failure label."""
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/events.csv")
OUT = Path("data/processed/features.csv")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # parse ISO timestamps
    df = df.sort_values(["machine_id", "timestamp"])
    for col in ["sensor_1","sensor_2","sensor_3","sensor_4","sensor_5","load","ambient_temp"]:
        df[f"{col}_roll3_mean"] = df.groupby("machine_id")[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
        df[f"{col}_roll3_std"] = df.groupby("machine_id")[col].transform(lambda s: s.rolling(3, min_periods=1).std().fillna(0))
        df[f"{col}_delta"] = df.groupby("machine_id")[col].diff().fillna(0)
    # failure within next hour (4 * 15-minute steps)
    df["failure_next_1h"] = (
        df.groupby("machine_id")["failure"].shift(-4).fillna(0).astype(int)
    )
    return df

def main():
    df = pd.read_csv(RAW)
    feat = build_features(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(OUT, index=False)
    print(f"Wrote features to {OUT}")

if __name__ == "__main__":
    main()
