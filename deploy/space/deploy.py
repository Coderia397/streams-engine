#!/usr/bin/env python3
"""Uploads the engine binary and the launcher to the Hugging Face Space. Used by the GitHub workflow; can be run by hand.
  HF_TOKEN=... python3 deploy/space/deploy.py <path to the streams-engine binary>"""
import os, pathlib, shutil, sys, tempfile
from huggingface_hub import HfApi

SPACE = os.environ.get("SPACE_REPO", "ibrahimar397/streams-engine")
here = pathlib.Path(__file__).resolve().parent
binary = pathlib.Path(sys.argv[1])
with tempfile.TemporaryDirectory() as tmp:
    for f in ("README.md", "requirements.txt", "app.py"):
        shutil.copy(here / f, tmp)
    shutil.copy(binary, pathlib.Path(tmp) / "streams-engine")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(repo_id=SPACE, repo_type="space", folder_path=tmp, commit_message=os.environ.get("COMMIT_MESSAGE", "Deploy the engine"))
print("deployed to", SPACE)
