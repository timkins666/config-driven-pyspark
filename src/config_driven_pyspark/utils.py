"""utility functions"""

from pyspark.sql import DataFrame


def flatten_schema(df: DataFrame) -> list[str]:
    """
    Flatten a dataframe schema.
    Returns a list of strings.
    Array type fields are denoted by [].
    """

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

    return flatten_struct(df.schema.jsonValue())


def limit_depth(field: str, depth: int | str = -1):
    """
    Limit a nested field path to `depth`.
    `depth = -1` means return up to the final parent.
    If `depth` is a string, will limit `field`'s depth to match.
    """

    split = field.split(".")

    if depth == -1:
        depth = len(split) - 1
    elif isinstance(depth, str):
        depth = depth.count(".") + 1

    return ".".join(field.split(".")[:depth])
