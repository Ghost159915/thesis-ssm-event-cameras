"""Stage-5 harness: put code+RVT on sys.path, register the drop-in backbone, and Hydra-compose the
REAL RVT train config with smoke overrides. Importing this module is side-effect-light; call the
functions explicitly."""
import os, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]
RVT = REPO / "external/ssms_event_cameras/RVT"


def setup_paths():
    for p in (REPO / "code", RVT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def register():
    from event_ssm.integration.register import register_resnet_mamba
    register_resnet_mamba()


def compose_smoke_config(dataset_path=None, max_epochs=50, batch_size=2, extra_overrides=None):
    setup_paths()
    register()
    os.environ.setdefault("WANDB_MODE", "disabled")
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from config.modifier import dynamically_modify_train_config
    if dataset_path is None:
        dataset_path = REPO / "data/gen1_smoke"
    overrides = [
        "dataset=gen1",
        "model=rnndet",                       # provides model.name=rnndet (modifier dispatches on it)
        "+experiment/gen1=resnet_mamba",      # pulls in /model/resnet_mamba_yolox + train/dataset blocks
        f"dataset.path={dataset_path}",
        "dataset.train.sampling=random",
        f"batch_size.train={batch_size}",
        f"batch_size.eval={batch_size}",
        "hardware.gpus=0",
        "hardware.num_workers.train=0",
        "hardware.num_workers.eval=0",
        f"training.max_epochs={max_epochs}",
    ] + (extra_overrides or [])
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(RVT / "config"), version_base="1.2"):
        cfg = compose(config_name="train", overrides=overrides)
    dynamically_modify_train_config(cfg)
    return cfg


if __name__ == "__main__":
    cfg = compose_smoke_config()
    print("composed OK:",
          "backbone=", cfg.model.backbone.name,
          "num_classes=", cfg.model.head.num_classes,
          "in_ch=", cfg.model.backbone.input_channels,
          "dataset.path=", cfg.dataset.path)
