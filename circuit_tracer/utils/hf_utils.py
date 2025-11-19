from __future__ import annotations

from typing import Dict, Iterable, NamedTuple, Optional
from urllib.parse import parse_qs, urlparse

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_ENABLE_HF_TRANSFER
from huggingface_hub.utils.tqdm import tqdm as hf_tqdm
from tqdm.contrib.concurrent import thread_map


class HfUri(NamedTuple):
    """Structured representation of a HuggingFace URI."""

    repo_id: str
    file_path: str
    revision: Optional[str]


def parse_hf_uri(uri: str) -> HfUri:
    """Parse an HF URI into repo id, file path and revision.

    Args:
        uri: String like ``hf://org/repo/file?revision=main``.

    Returns:
        ``HfUri`` with repository id, file path and optional revision.
    """
    parsed = urlparse(uri)
    if parsed.scheme != "hf":
        raise ValueError(f"Not a huggingface URI: {uri}")
    path = parsed.path.lstrip("/")
    repo_parts = path.split("/", 1)
    if len(repo_parts) != 2:
        raise ValueError(f"Invalid huggingface URI: {uri}")
    repo_id = f"{parsed.netloc}/{repo_parts[0]}"
    file_path = repo_parts[1]
    revision = parse_qs(parsed.query).get("revision", [None])[0] or None
    return HfUri(repo_id, file_path, revision)


def download_hf_uri(uri: str) -> str:
    """Download a file referenced by a HuggingFace URI and return the local path."""
    parsed = parse_hf_uri(uri)
    return hf_hub_download(
        repo_id=parsed.repo_id,
        filename=parsed.file_path,
        revision=parsed.revision,
        force_download=False,
    )


def download_hf_uris(uris: Iterable[str], max_workers: int = 8, local_dir: Optional[str] = None) -> Dict[str, str]:
    """Download multiple HuggingFace URIs concurrently.

    Args:
        uris: Iterable of HF URIs.
        max_workers: Maximum number of parallel workers when HF transfer is
            disabled. Ignored otherwise.

    Returns:
        Mapping from input URI to the local file path on disk.
    """
    if not uris:
        return {}

    parsed_map = {uri: parse_hf_uri(uri) for uri in uris}

    def _download(uri: str) -> str:
        info = parsed_map[uri]
        # return hf_hub_download(
        #     repo_id=info.repo_id,
        #     filename=info.file_path,
        #     revision=info.revision,
        #     force_download=False,
        #     local_dir_use_symlinks=False, 
        # )
        download_kwargs = dict(
            repo_id=info.repo_id,
            filename=info.file_path,
            revision=info.revision,
            force_download=False,
            local_dir_use_symlinks=False,
        )
        if local_dir is not None:
            download_kwargs["local_dir"] = local_dir
        return hf_hub_download(**download_kwargs)
    
    if HF_HUB_ENABLE_HF_TRANSFER:
        return {uri: _download(uri) for uri in uris}

    results = thread_map(
        _download,
        list(uris),
        desc=f"Fetching {len(parsed_map)} files",
        max_workers=max_workers,
        tqdm_class=hf_tqdm,
    )
    return dict(zip(uris, results))
