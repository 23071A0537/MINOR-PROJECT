#!/usr/bin/env python
import importlib
import sys


def safe_version(pkg_name):
    try:
        mod = importlib.import_module(pkg_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def main():
    print("=" * 64)
    print("GPU stack verification")
    print("=" * 64)
    print(f"Python: {sys.version.split()[0]}")

    torch_version = safe_version("torch")
    tf_version = safe_version("tensorflow")
    xgb_version = safe_version("xgboost")
    qml_version = safe_version("pennylane")
    jax_version = safe_version("jax")

    print(f"torch: {torch_version or 'not installed'}")
    print(f"tensorflow: {tf_version or 'not installed'}")
    print(f"xgboost: {xgb_version or 'not installed'}")
    print(f"pennylane: {qml_version or 'not installed'}")
    print(f"jax: {jax_version or 'not installed'}")

    gpu_ready = False

    if torch_version:
        import torch

        print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
        print(f"torch CUDA runtime: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU 0: {torch.cuda.get_device_name(0)}")
            gpu_ready = True

    if tf_version:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        print(f"tensorflow GPUs: {len(gpus)}")

    if xgb_version:
        try:
            import xgboost as xgb
            import numpy as np

            X = np.random.rand(128, 8).astype("float32")
            y = (X[:, 0] > 0.5).astype("int32")
            model = xgb.XGBClassifier(
                n_estimators=5,
                max_depth=3,
                learning_rate=0.1,
                tree_method="hist",
                device="cuda",
                verbosity=0,
            )
            model.fit(X, y)
            print("xgboost CUDA check: OK")
            gpu_ready = True or gpu_ready
        except Exception as exc:
            print(f"xgboost CUDA check: failed ({exc})")

    print("=" * 64)
    if gpu_ready:
        print("GPU acceleration is available for this project on this machine.")
        return 0

    print("No GPU backend was confirmed. PyTorch CUDA setup may be incomplete.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
