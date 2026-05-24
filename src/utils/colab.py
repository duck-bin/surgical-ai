"""Google Colab helpers.

Importable everywhere -- the ``google.colab`` dependency is imported lazily
inside :func:`mount_drive`, so this module loads fine off Colab (and in CI).
"""
from __future__ import annotations

import os
from pathlib import Path


def mount_drive(project: str = "surgical-cvs-ai",
                mount_point: str = "/content/drive",
                link=("data", "outputs")) -> str:
    """Mount Google Drive and persist datasets / checkpoints / model cache.

    Colab wipes ``/content`` on a runtime reset, so re-running re-downloads the
    multi-GB datasets and loses training checkpoints. This:

    - mounts Drive at ``mount_point`` (prompts for auth on first run);
    - symlinks the working-dir ``data/`` and ``outputs/`` to a Drive project
      folder, so downloads and checkpoints land on Drive and survive resets;
    - points ``HF_HOME`` at Drive so HuggingFace model downloads (SAM2,
      ViT, ...) are cached there too.

    Call it once, early -- before importing ``transformers`` -- so ``HF_HOME``
    takes effect. Returns the Drive project path.
    """
    try:
        from google.colab import drive  # noqa: PLC0415 (Colab-only, lazy)
    except ImportError as exc:
        raise RuntimeError("mount_drive() runs only on Google Colab.") from exc

    drive.mount(mount_point)
    base = Path(mount_point) / "MyDrive" / project
    (base / "hf_cache").mkdir(parents=True, exist_ok=True)

    # Persist HuggingFace model downloads (SAM2, ViT-Small, ...) on Drive.
    os.environ["HF_HOME"] = str(base / "hf_cache")

    for name in link:
        target = base / name
        target.mkdir(parents=True, exist_ok=True)
        local = Path(name)
        if local.is_symlink():
            local.unlink()
        elif local.is_dir():
            if any(local.iterdir()):
                print(f"[drive] '{name}/' already has local files; not linking.")
                continue
            local.rmdir()
        local.symlink_to(target)
        print(f"[drive] {name}/ -> {target}")

    print("[drive] HF_HOME ->", os.environ["HF_HOME"])
    return str(base)
