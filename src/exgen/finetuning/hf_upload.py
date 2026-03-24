import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.create_repo(repo_id="Whomstt/exgen-qlora-adapter", repo_type="model", exist_ok=True)
api.upload_large_folder(
    folder_path="exgen-qlora-adapter",
    repo_id="Whomstt/exgen-qlora-adapter",
    repo_type="model",
)
