# download_models.py

import os
from huggingface_hub import hf_hub_download, list_repo_files

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained_models", "smplest_x")
SMPLX_DIR      = os.path.join(BASE_DIR, "human_models", "human_model_files", "smplx")
SMPL_DIR       = os.path.join(BASE_DIR, "human_models", "human_model_files", "smpl")

def download_smplestx():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    print("[✓] Downloading SMPLest-X weights...")

    hf_hub_download(
        repo_id="waanqii/SMPLest-X",
        filename="smplest_x_h.pth.tar",
        local_dir=PRETRAINED_DIR
    )

    hf_hub_download(
        repo_id="waanqii/SMPLest-X",
        filename="config_base.py",
        local_dir=PRETRAINED_DIR
    )

def download_parametric_models():
    repo_id = "camenduru/SMPLer-X"
    files = list_repo_files(repo_id)

    smplx_files = [f for f in files if f.startswith(("SMPLX", "SMPL-X", "MANO"))]
    smpl_files  = [f for f in files if f.startswith("SMPL_")]

    # SMPLX / MANO files → smplx/
    os.makedirs(SMPLX_DIR, exist_ok=True)
    print(f"[✓] Downloading {len(smplx_files)} files to smplx/")
    for f in smplx_files:
        hf_hub_download(repo_id=repo_id, filename=f, local_dir=SMPLX_DIR)

    # SMPL files → smpl/
    os.makedirs(SMPL_DIR, exist_ok=True)
    print(f"[✓] Downloading {len(smpl_files)} files to smpl/")
    for f in smpl_files:
        hf_hub_download(repo_id=repo_id, filename=f, local_dir=SMPL_DIR)

if __name__ == "__main__":
    download_smplestx()
    download_parametric_models()
    print("\n[✓] All model files downloaded successfully.")
