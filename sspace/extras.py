"""Compatibility entry point; use sspace.experiment_steps."""
from sspace.experiment_steps import ENTRYPOINTS as EXTRAS, main

__all__ = ("EXTRAS", "main")

if __name__ == "__main__":
    main()
