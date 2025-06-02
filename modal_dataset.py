import modal

app = modal.App("vit-fairface-dataset")
volume = modal.Volume.from_name("fairface-data", create_if_missing=True)

@app.function(
    image=modal.Image.debian_slim().pip_install("datasets", "pandas", "pyarrow"),
    volumes={"/data": volume},
    timeout=900
)
def download_fairface():
    from datasets import load_dataset

    print("📥 Loading full FairFace dataset...")
    dataset = load_dataset("HuggingFaceM4/FairFace", data_dir="1.25")

    print("💾 Saving training set...")
    dataset["train"].to_parquet("/data/train.parquet")

    print("💾 Saving validation set...")
    dataset["validation"].to_parquet("/data/validation.parquet")

    print("✅ All splits saved to volume!")

@app.local_entrypoint()
def main():
    download_fairface.remote()
