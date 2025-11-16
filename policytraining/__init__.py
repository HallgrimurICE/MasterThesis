"""Top-level package for the policy training utilities used in Diplomacy experiments."""

def package_root() -> str:
    """Return the absolute path to the policytraining package.

    Useful in ad-hoc scripts where we need to locate assets relative to the
    repository root without assuming installation as a site package.
    """
    from pathlib import Path

    return str(Path(__file__).resolve().parent)
