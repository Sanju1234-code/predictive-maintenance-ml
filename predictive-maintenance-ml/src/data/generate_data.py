"""Generate synthetic raw sensor events for predictive maintenance."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def main(out_path: str = "data/raw/events.csv", seed: int = 42):
    rng = np.random.default_rng(seed)
    n_machines = 120
    rows_per_machine = 120  # 15-min interval ≈ 30 hours
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    records = []
    for m in range(n_machines):
        base = rng.normal(0, 1, size=5)
        drift = rng.normal(0, 0.02, size=5).cumsum()
        timestamps = [start_time + timedelta(minutes=15*i) for i in range(rows_per_machine)]
        health = 1 - (np.arange(rows_per_machine) / (rows_per_machine * rng.uniform(1.1, 1.6)))
        health += rng.normal(0, 0.03, size=rows_per_machine)
        spikes_idx = rng.choice(rows_per_machine, size=rng.integers(2, 6), replace=False)
        spikes = np.zeros(rows_per_machine)
        spikes[spikes_idx] = rng.normal(0.6, 0.2, size=len(spikes_idx))
        for i, ts in enumerate(timestamps):
            sensors = base + drift + rng.normal(0, 0.3, size=5) + spikes[i]
            p_fail = np.clip(0.02 + (0.55 * (1 - health[i])) + (0.35 * (spikes[i] > 0)), 0, 0.9)
            failure = rng.random() < p_fail
            records.append({
                "machine_id": f"M{m:03d}",
                "timestamp": ts.isoformat(),
                "sensor_1": sensors[0],
                "sensor_2": sensors[1],
                "sensor_3": sensors[2],
                "sensor_4": sensors[3],
                "sensor_5": sensors[4],
                "ambient_temp": 25 + rng.normal(0, 1),
                "load": np.clip(rng.normal(0.65, 0.15), 0, 1.2),
                "failure": int(failure)
            })

    df = pd.DataFrame.from_records(records).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows to {out}")

if __name__ == "__main__":
    main()
