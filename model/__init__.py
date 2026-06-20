__all__ = ["SkateFormer", "SkateFormerPre", "STGCN"]


def __getattr__(name):
    if name == "SkateFormer":
        from .SkateFormer import Model
    elif name == "SkateFormerPre":
        from .SkateFormerPre import Model
    elif name == "STGCN":
        from .st_gcn import Model
    else:
        raise AttributeError(f"module 'model' has no attribute {name!r}")
    return Model
