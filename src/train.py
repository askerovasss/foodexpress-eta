import argparse, os, joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from features import build_frame

def main(input_csv: str, model_out: str):
    df = pd.read_csv(input_csv)
    X_raw, y, num_cols, cat_cols, _ = build_frame(df)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_tr, X_val, y_tr, y_val = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    rmse = mean_squared_error(y_val, preds, squared=False)
    r2 = r2_score(y_val, preds)
    print(f"Validation -> MAE={mae:.3f}; RMSE={rmse:.3f}; R2={r2:.3f}")

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(pipe, model_out)
    print(f"Saved pipeline to {model_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--model", default="artifacts/model.joblib")
    args = p.parse_args()
    main(args.input, args.model)
