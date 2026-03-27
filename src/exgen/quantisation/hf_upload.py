import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

api = HfApi(token=os.getenv("HF_TOKEN"))
api.create_repo(
    repo_id="Whomstt/Ministral-3-3B-Instruct-2512-bnb-nf4",
    repo_type="model",
    exist_ok=True,
)
api.upload_large_folder(
    folder_path="Ministral-3-3B-Instruct-2512-bnb-nf4",
    repo_id="Whomstt/Ministral-3-3B-Instruct-2512-bnb-nf4",
    repo_type="model",
)
