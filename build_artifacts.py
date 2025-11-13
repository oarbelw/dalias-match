import pickle
import pandas as pd
import backend.core_recommender as core
import os
import time

# Path to your local dataset
DATA_PATH = "/Users/orenarbel-wood/Desktop/Dovie/ratings_df.parquet"
OUT_PATH = "/Users/orenarbel-wood/Desktop/Dovie/artifacts.pkl"

def main():
    start = time.time()
    print(f"🔹 Loading dataset from {DATA_PATH} ...")

    # Load dataset using your core_recommender loader
    if DATA_PATH.endswith(".parquet"):
        df = pd.read_parquet(DATA_PATH)
    else:
        df = pd.read_csv(DATA_PATH)

    print(f"✅ Loaded dataset with shape {df.shape}")

    # Build the artifacts
    print("🔹 Building recommendation artifacts (this may take a few minutes)...")
    artifacts = core.build_model_artifacts(df)

    # Check that artifacts look valid
    if not isinstance(artifacts, dict) or len(artifacts) == 0:
        raise ValueError("❌ build_model_artifacts() returned an empty or invalid object.")

    # Safe-write with temporary file to prevent partial writes
    temp_path = OUT_PATH + ".tmp"
    with open(temp_path, "wb") as f:
        pickle.dump(artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(temp_path, OUT_PATH)
    print(f"✅ Saved {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)")
    print(f"⏱️ Total time: {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    main()
