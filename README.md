# Config driven PySpark

A tool for manipulating dataframes with nested structures using preset configs, e.g. from a YAML file.

This is presented as a guide to hopefully save you a headache, feel free to take it and repurpose however you wish.

## Why

For "flat" dataframes where all data is in primitive-type root columns, manipulating the data is a trival task. However, things get more complicated and verbose for nested data structures, where multiple chained `withField` operations per root column are often required. Further complexity is added by arrays, where the `transform` PySpark function must be used to apply functions to the array elements rather than the array itself. Manually writing these quickly becomes unfeasible.

## Usage

Simply create an instance of the `Functioniser` and call the `add` method for each dataframe field to be transformed with the field name and the function to call on the data. Each field can be limitlessly nested and inside limitless arrays within the data structure (as far as this tool is concerned).

Field names are added as their flattened path, and you don't need to tell it which fields are arrays; it will determine array types from the dataframe schema.

Functions can be added either:
- as a string to call PySpark builtins (from `pyspark.sql.functions`) or [custom registered](#registering-custom-functions) functions by name (see below)
- as a PySpark `Column`, e.g. to use a builtin like `pyspark.sql.functions.lit`
- as a function taking a single `Column` parameter and return a `Column`

Once all fields are added, call `Functioniser.apply` with your dataframe. That's it.

An example manual implementation would be:

```python
from pyspark.sql import functions as F

runner = Functioniser()

runner = (
    Functioniser()
    .add("some_root", "upper")
    .add("another_root.nested_field", F.lit("foo"))
    .add(
        "another_root.another_nested_field",
        # this is contrived nonsense, but illustrative of what you can do
        lambda col: F.concat(
            F.split_part("some_root", F.lit("/"), F.lit(3)), col
        ),
    )
)

modified_df = runner.apply(df)
```

### Config driven

The config format is really up to you. As long as you can load it, iterate through the result and `add` the field and functions to apply then that's all you need. You can even inspect your dataframe for certain fields or types and use those to create your configs on the fly, instead of or in addition to using static config files.

#### Registering custom functions

In addition to using PySpark functions by name in `Functioniser.add`, you can use `Functioniser.register_function` to set up additional "known" functions that can be referred to by their name in static config files. This alleviates the need for an in-code mapping between functions in Python code and their in the configs.

### Example config driven usage

```yaml
functions:
  my_root.field_a: trim
  my_root.field_b: substr_first_four
  my_root.field_c: set_to_foo
```

```python
from pyspark.sql import functions as F

runner = Functioniser()

runner.register_function("substr_first_four", lambda col: F.substring(col, 0, 4))
runner.register_function("set_to_foo", F.lit("foo"))

for field, fn in parsed_yaml["functions"].items():
    runner.add(field, fn)
```

## What it does

Under the hood, it takes your configured field names and builds a nested structure "plan" respresenting all struct nodes from root columns down to the primitive-type leaf fields we'll be applying the functions to. This allows us to build the final dataframe using a single `select` operation on the original dataframe (using a `withColumn` in a loop is a. generally bad, and b. very bad here as the query plans will be enormous and either very ineffecient or just break Spark).

It then iterates recursively through the plan, using a `Column` object to represent the position in the structure (the "context"). The initial context is the root field, and chained `withField` operations are added for each context level.

If the current context is a struct, it uses `getField` to set the next context to member fields as it recurses deeper into the schema.

If the context is an array, it uses `transform` to operate on each element of the array, and the array elements become the context in the next call.

If the context is the leaf field we're applying functions to, it returns the configured Spark function called with context as the `Column` parameter.

## Extending it

### Nested data select function

If you `select` nested fields from a dataframe, the result is a flat dataframe with all selected leaf fields as root columns. It also doesn't play very nicely (or at all) with arrays in nested structures, particularly through multiple levels of array.

The alternative is to apply the `pyspark.sql.functions.drop` function to structs to prune the fields you don't want, which maintains the original structure. There are some shenanigans involved in determining where and what to drop, but the results can be hooked up using the `Functioniser.add` method.

### Case senitivity

It operates case-insensitively so that the casing of config files doesn't have to match that of the dataframe. This can, of course, be adapted if you need it.
