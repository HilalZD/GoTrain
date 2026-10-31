"""Training engines. Use `get_engine()`; never import a concrete engine directly."""

from gotrain.engine.interface import Engine, get_engine, load_band_stats

__all__ = ["Engine", "get_engine", "load_band_stats"]
