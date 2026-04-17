"""Phase A+B meshfree shape function package."""

from typing import Any

__all__ = ["ShapeFunctionModel"]


def __getattr__(name: str) -> Any:
    if name == "ShapeFunctionModel":
        from .models.full_model import ShapeFunctionModel

        return ShapeFunctionModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
