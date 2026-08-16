from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="photonmz/pdbbindpp-2020",
    repo_type="dataset",
    local_dir="data/refined_set"
)