from typing import Any, Callable, Iterable
from pyspark.sql import DataFrame


def flatten_schema(df: DataFrame):
    def flatten_struct(struct, prefix=""):
        result = []

        if struct["type"] == "array":
            prefix += "[]"
            if isinstance(struct["elementType"], str):
                result.append(prefix)
            else:
                result += flatten_struct(struct["elementType"], prefix)
        else:
            for field in struct["fields"]:
                field_name = f"{prefix}." * bool(prefix) + field["name"]
                if isinstance(field["type"], dict):
                    result += flatten_struct(field["type"], field_name)
                else:
                    result.append(field_name)
        return result

    flat = flatten_struct(df.schema.jsonValue())
    return flat


def limit_depth(field: str, depth=-1):
    """
    limit a nested field path to `depth`
    depth = -1 means return up to the final parent
    """

    split = field.split(".")

    if depth == -1:
        depth = len(split) - 1

    return ".".join(field.split(".")[:depth])


def find_first(l: Iterable, predicate: Callable[[Any], bool]):
    for i, x in enumerate(l):
        if predicate(x):
            return x, i
    raise ValueError("No element found")
