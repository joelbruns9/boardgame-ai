"""NNUE evaluation pipeline for the alpha-beta searcher.

The dense Step-2 path remains in ``net.py``/``data.py``/``train.py``. Step 3 adds
the generic sparse network in ``sparse_net.py`` and Kingdomino-specific replay,
feature, and target adapters in ``sparse_data.py``. A second game can reuse the
sparse architecture and training math while supplying its own packed-data adapter.
"""
