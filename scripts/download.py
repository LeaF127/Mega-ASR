import argparse
from pathlib import Path

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ImportError:  # pragma: no cover - handled at runtime
    snapshot_download = None


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "zhifeixie/Mega-ASR"
DEFAULT_LOCAL_DIR = ROOT_DIR / "ckpt/Mega-ASR"


def parse_args():
    parser = argparse.ArgumentParser(description="Download Mega-ASR weights from ModelScope")
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID, help="ModelScope model id")
    parser.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR, help="local ckpt dir")
    return parser.parse_args()


def main():
    args = parse_args()
    if snapshot_download is None:
        raise RuntimeError("Please install modelscope first: pip install modelscope")

    snapshot_download(
        model_id=args.repo_id,
        local_dir=str(args.local_dir),
    )
    print(f"Downloaded to {args.local_dir}")


if __name__ == "__main__":
    main()
