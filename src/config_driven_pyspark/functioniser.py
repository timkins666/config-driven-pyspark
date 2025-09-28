"""A handy thing to apply functions to nested data"""

from collections import UserDict
import inspect
import logging
from typing import Callable
from pyspark.sql import Column, DataFrame, functions as F
import utils

type DfFunction = Callable[[Column], Column]

_logger = logging.getLogger(__name__)


class Functioniser:
    """Applies functions to fields in the dataframe"""

    _spark_fns: dict[str, Column | DfFunction] = {
        f[0]: f[1]
        for f in inspect.getmembers(F, inspect.isfunction)
        if not f[0].startswith("_")
    }

    def __init__(self):
        self.reset()
        self._custom_functions: dict[str, Column | DfFunction] = {}

    def reset(self) -> None:
        """Reset to start afresh to run with a new DataFrame"""
        self.functions: dict[str, DfFunction] = {}
        self.flat_schema: list[str] = []

    def add(self, field: str, function: str | Column | DfFunction) -> "Functioniser":
        """Add a function to be applied to the DataFrame"""
        field = field.lower()

        if isinstance(function, str):
            if function in self._custom_functions:
                function = self._custom_functions[function]
            elif function in self._spark_fns:
                function = self._spark_fns[function]
            else:
                raise ValueError(f"Function {function} not found")

        fn: DfFunction = (
            (lambda _: function) if isinstance(function, Column) else function
        )

        if field in self.functions:
            existing_fn = self.functions[field]
            self.functions[field] = lambda col: fn(existing_fn(col))
        else:
            self.functions[field] = fn

        return self

    def apply(self, df: DataFrame) -> DataFrame:
        """Apply the configured functions to the DataFrame"""
        self.flat_schema = utils.flatten_schema(df)
        node_map = self._build_nodes()

        return df.select(
            *[
                self._apply(F.col(root_col), node_map.iget(root_col)).alias(root_col)
                for root_col in df.columns
            ]
        )

    def _apply(self, ctx: Column, node: "NodeFunctions | None"):
        """
        Iterate through members of `node` and apply any configured functions
        to lead nodes.
        """
        # no configs for this root, just return
        if node is None:
            return ctx

        # apply function to current ctx
        if node.node_function:
            if node.is_array:
                return F.transform(ctx, node.node_function)
            return node.node_function(ctx)

        for member in node:
            if node.is_array:
                ctx = F.transform(ctx, self._create_array_transform(member, node))
            else:
                ctx = ctx.withField(
                    member,
                    self._apply(
                        ctx.getField(member),
                        node[member],
                    ),
                )

        return ctx

    def _create_array_transform(self, member: str, node: "NodeFunctions"):
        """Create array transform function with closure"""

        def transform_fn(element):
            return element.withField(
                member,
                self._apply(
                    element.getField(member),
                    node[member],
                ),
            )

        return transform_fn

    def _build_nodes(self):
        root_map = NodeFunctions("root")

        for field, function in self.functions.items():
            current = root_map

            try:
                field_with_tokens = next(
                    f
                    for f in self.flat_schema
                    if f.lower().replace("[]", "") == field.lower()
                )
            except StopIteration:
                _logger.warning("Field %s not found in schema", field)
                continue

            split = field_with_tokens.split(".")
            for i, member in enumerate(split):
                member_clean = member.replace("[]", "")
                if member_clean not in current:
                    current[member_clean] = NodeFunctions(
                        member,
                        is_array=member.endswith("[]"),
                        function=function if i == len(split) - 1 else None,
                    )
                current = current[member_clean]
            current.node_function = self.functions[field]
        return root_map

    def register_function(
        self, name: str, func: Column | Callable[[Column], Column]
    ) -> None:
        """
        Register a custom function to use by name.
        Can be used to override default Spark functions.
        """
        if not (isinstance(func, Column) or callable(func)):
            raise ValueError(f"Function {name} must be a Column or callable")
        self._custom_functions[name] = func


class NodeFunctions(UserDict[str, "NodeFunctions"]):
    """dict of member names to:
    - another NodeFunctions instance for struct members
    - or a function to apply to the column/field
    """

    def __init__(
        self,
        node_name: str,
        *,
        is_array=False,
        function: Callable | None = None,
    ):
        super().__init__()

        self.name = node_name.replace("[]", "")
        self.is_array = is_array
        self.node_function = function

    def iget(self, key: str):
        """Get value by case insensitive key"""

        matched_key = next((x for x in self if x.lower() == key.lower()), None)
        return self[matched_key] if matched_key is not None else None
