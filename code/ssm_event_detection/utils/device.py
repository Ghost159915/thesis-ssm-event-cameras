"""Device detection utility — handles CUDA, ROCm, MPS, and CPU fallback."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return the best available torch device.

    Args:
        preference: One of "auto", "cuda", "rocm", "mps", "cpu".
            "auto" tries CUDA/ROCm first, then MPS, then CPU.

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            print(f"[device] Using CUDA: {name}")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[device] Using Apple MPS")
            return torch.device("mps")
        else:
            print("[device] Using CPU")
            return torch.device("cpu")
    else:
        device = torch.device(preference)
        print(f"[device] Using: {device}")
        return device
