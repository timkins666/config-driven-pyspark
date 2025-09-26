import logging
from typing import Callable
from pyspark.sql import Column, DataFrame, functions as F

import utils

type DfFunction = Callable[[Column], Column]

_logger = logging.getLogger(__name__)


class DataframeFunctions:

    def __init__(self):
        self.reset()

    def reset(self):
        self.functions: dict[str, DfFunction] = {}
        self.flat_schema: list[str] = []

    def add(self, field: str, function: Column | DfFunction):
        field = field.lower()

        fn: DfFunction = (
            (lambda _: function) if isinstance(function, Column) else function
        )

        if field in self.functions:
            self.functions[field] = lambda col: fn(self.functions[field](col))
        else:
            self.functions[field] = fn

    def run(self, df: DataFrame):
        self.flat_schema = utils.flatten_schema(df)
        # xxx use select
        for field, function in self.functions.items():
            try:
                name_with_tokens = next(
                    (
                        x
                        for x in self.flat_schema
                        if x.lower().replace("[]", "") == field.lower()
                    )
                )
            except StopIteration:
                _logger.warning("Field %s not found in schema", field)
                continue

            df = df.withColumn(
                utils.limit_depth(field, 1),
                self._apply_root(name_with_tokens, function),
            )

        return df

    def _apply_root(self, field: str, function: DfFunction):
        if "." not in field:
            return self._apply_root_column(field, function)

        root_clean = utils.limit_depth(field, 1).replace("[]", "")
        is_root_array = utils.limit_depth(field, 1).endswith("[]")
        root_ctx = F.col(root_clean)

        if not is_root_array:
            return root_ctx.withField(
                field.split(".", 1)[1].replace("[]", ""),
                self._apply_function(field.split(".", 1)[1], root_ctx, function),
            )

        return F.transform(root_ctx, lambda element: element.withField(
            field.split(".", 1)[1].replace("[]", ""),
            self._apply_function(field.split(".", 1)[1], element, function),
        ))

    def _apply_function(self, field: str, ctx: Column, function: DfFunction):
        if "[]" not in field:
            return function(ctx if not field else ctx.getField(field))

        split = field.split(".")
        _, first_array_idx = utils.find_first(split, lambda x: x.endswith("[]"))

        array = ctx.getField(".".join(split[: first_array_idx + 1]).replace("[]", ""))

        return F.transform(
            array,
            lambda element: self._apply_function(
                ".".join(split[first_array_idx + 1 :]), element, function
            ),
        )

    def _apply_root_column(self, root_name: str, function: DfFunction):
        if "[]" not in root_name:
            return function(F.col(root_name))
        return F.transform(root_name.removesuffix("[]"), function)
