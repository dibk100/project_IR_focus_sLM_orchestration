from huggingface_hub import snapshot_download

models = [
    "codellama/CodeLlama-7b-Instruct-hf",
    # "Qwen/Qwen2.5-Coder-14B-Instruct",
    # "meta-llama/Llama-3.1-8B-Instruct"
]

for repo_id in models:
    print(f"Downloading {repo_id} ...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=None,              # None → HF_HOME 경로 사용
        local_dir_use_symlinks=False,
        resume_download=True
    )

print("Download completed.")

# 저장 위치 확인 방법
# ls /mnt/hdd/hf_cache/hub/models--*
# os.environ["HF_HOME"] = "/mnt/hdd/hf_cache"