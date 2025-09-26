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
