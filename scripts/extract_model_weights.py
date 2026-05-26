#!/usr/bin/env python3
"""Extract the model ``state_dict`` from a Lightning ``.ckpt``.

A Lightning checkpoint produced by ``nequip-train`` contains the full
optimizer / scheduler / EMA state in addition to the model weights. For
stage-to-stage transfer in the staged training schedule
(``experiments/5-composite_staged``) we only need the model weights
themselves, exposed as a flat ``{name: tensor}`` mapping with the
stage's stripped key prefix.

Usage::

    python scripts/extract_model_weights.py INPUT.ckpt OUTPUT.pt \\
        [--strip-prefix model.sole_model.]

  * ``INPUT.ckpt``: path to a Lightning checkpoint (``best.ckpt`` etc.).
  * ``OUTPUT.pt``: destination file. Overwrites if it exists.
  * ``--strip-prefix``: prefix removed from every key. Default
    ``model.sole_model.`` matches the wrapping that
    ``NequIPLightningModule`` applies (``model`` is a ``ModuleDict``
    with the sole entry under the key ``sole_model``). After stripping,
    keys look like ``m0_radial_chemical_embed.type_embed.weight`` —
    exactly what the bare model's ``load_state_dict`` consumes via
    :class:`sparse_delta_core.train.LoadWeightsCallback`.

The output ``.pt`` is a flat ``Dict[str, Tensor]`` saved with
``torch.save``. Loading via
``torch.load(weights_only=True)`` is safe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Extract a Lightning checkpoint's model state_dict."
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to a Lightning checkpoint (e.g. best.ckpt).",
    )
    p.add_argument(
        "output",
        type=Path,
        help="Path to write the extracted state_dict (.pt).",
    )
    p.add_argument(
        "--strip-prefix",
        default="model.sole_model.",
        help=(
            "Key prefix removed from every entry. Default "
            "'model.sole_model.' matches NequIPLightningModule's "
            "ModuleDict wrapping."
        ),
    )
    p.add_argument(
        "--keep-prefix",
        default=None,
        help=(
            "If set, only keys starting with this prefix (BEFORE "
            "stripping) are emitted. Use to extract a sub-block "
            "(e.g. '--keep-prefix model.sole_model.m0_')."
        ),
    )
    args = p.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(
        args.input, map_location="cpu", weights_only=False
    )
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        print(
            f"error: {args.input} does not look like a Lightning "
            f"checkpoint (no 'state_dict' key).",
            file=sys.stderr,
        )
        return 3

    raw = ckpt["state_dict"]
    out: dict[str, torch.Tensor] = {}
    skipped_no_prefix = 0
    skipped_filtered = 0
    for k, v in raw.items():
        if args.keep_prefix is not None and not k.startswith(
            args.keep_prefix
        ):
            skipped_filtered += 1
            continue
        if args.strip_prefix and not k.startswith(args.strip_prefix):
            skipped_no_prefix += 1
            continue
        new_k = k[len(args.strip_prefix) :] if args.strip_prefix else k
        out[new_k] = v.detach().cpu()

    if not out:
        print(
            f"error: zero keys matched. Raw checkpoint had "
            f"{len(raw)} keys; first 3: {list(raw)[:3]}",
            file=sys.stderr,
        )
        return 4

    torch.save(out, args.output)
    print(
        f"wrote {len(out)} parameters to {args.output} "
        f"(skipped_no_prefix={skipped_no_prefix}, "
        f"skipped_filtered={skipped_filtered}). "
        f"first 3 keys: {list(out)[:3]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
