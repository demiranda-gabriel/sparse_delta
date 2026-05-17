"""Smoke test: the warm-start composite config Hydra-instantiates.

Run with::

    uv run python experiments/4-composite_warmstart/test_config_instantiate.py

No GPU or training data required. Catches config drift early (a
renamed kwarg or a typo in the ``_target_`` blows up here long before
``sbatch``).
"""
from __future__ import annotations
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict
from nequip.utils.global_state import set_global_state


def main() -> int:
    set_global_state(allow_tf32=False)
    cfg_dir = Path(__file__).resolve().parent
    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cfg = compose(config_name="config")

    # @model_builder requires seed, model_dtype, type_names — nequip-train
    # fills these in at training time. For the smoke instantiate, inject
    # them into the *inner* build_warmstart_composite kwargs (the outer
    # `nequip.model.modify` doesn't take them).
    # _recursive_=False so build_warmstart_composite instantiates its
    # nested radial_chemical_embed dict itself (after adding type_names).
    model_cfg = cfg.training_module.model
    inner = model_cfg.model if "model" in model_cfg else model_cfg
    with open_dict(inner):
        inner.seed = cfg.common_seed
        inner.model_dtype = "float32"
        # cuEquivariance modifier requires libcuda.so.1 only at forward
        # time; init is fine on a CPU login node. Disable PT2 compile
        # for the smoke so the lazy compile doesn't fire (the actual
        # training run re-enables it via the config).
        if "compile_mode" in inner:
            inner.compile_mode = "eager"
    # ``nequip.model.modify`` asserts modifiers is a plain list, not an
    # OmegaConf ListConfig, and the inner kwargs must be a plain dict
    # (otherwise the `**dict` unpacking yields ConfigType objects that
    # the @model_builder kwarg assertions reject).
    # ``_convert_="all"`` instructs hydra to convert every nested
    # OmegaConf node to native Python containers before the target call.
    model = instantiate(model_cfg, _recursive_=False, _convert_="all")
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] OK — built {type(model).__name__} with "
          f"{n_params} params ({n_train} trainable)")

    # Spot-check the warm-start aliasing.
    energy_net = model.model.func
    m0_advertised = energy_net.m0_allegro.irreps_out["_allegro_pre_final_tp_out"]
    m1_consumed = energy_net.m1_allegro.irreps_in["_allegro_pre_final_tp_out"]
    assert m0_advertised == m1_consumed, (
        f"M0 advertises {m0_advertised} but M1 consumes {m1_consumed}"
    )
    print(f"[smoke] OK — M1 aliasing matches M0 ({m0_advertised})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
