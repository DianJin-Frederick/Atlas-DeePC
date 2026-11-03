"""Read retained Reacher configurations without restoring retired options."""


def normalize_saved_config(values):
    """Drop the historical zero input-feature weight; never reinterpret nonzero weights."""
    config = dict(values)
    if config.pop("clustering_input_weight", 0.0) != 0.0:
        raise ValueError("Saved clustering_input_weight is nonzero; this is not joint-configuration-only clustering")
    return config
