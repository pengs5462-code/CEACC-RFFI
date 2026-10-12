"""Compatibility boundary for the superseded single-split data loader.

The final paper pipeline uses ``multiknown17_v1`` arrays and registered JSON
manifests, not the earlier loader. Some shared metric modules retain a legacy
import so that archived diagnostic runners remain readable. Any attempt to use
that old loader fails with a direct migration message instead of silently
changing the experimental protocol.
"""


def __getattr__(name):
    raise RuntimeError(
        f"legacy data-loader attribute {name!r} is not part of the public "
        "artifact; prepare multiknown17_v1 with src/prepare_multiknown17.py "
        "and use the registered multi-known runners"
    )
