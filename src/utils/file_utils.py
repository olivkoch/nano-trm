import shutil
from pathlib import Path


def download_directory(src: str, dst: Path):
    src_path = Path(src)
    if src_path.exists():
        if src_path.is_dir():
            shutil.copytree(src_path, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst)
    else:
        raise FileNotFoundError(f"Source path does not exist: {src}")


def upload_directory(src: Path, dst: str):
    src_path = Path(src)
    dst_path = Path(dst)
    if src_path.exists() and src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Source directory does not exist or is not a directory: {src}")
