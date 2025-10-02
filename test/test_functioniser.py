from pyspark.sql import SparkSession, functions as F
import pytest
from unittest import mock
import yaml

from functioniser import Functioniser
from test.conftest import to_df
import logging


class TestRootColumns:
    def _source_data(self, spark: SparkSession):
        data = [
            {
                "id": 1,
                "firstname": "Christopher",
                "lastname": "Lee",
                "unchanged": "yes",
                "rootarray": [1, 2],
            },
            {
                "id": 2,
                "firstname": "Miriam",
                "lastname": "Margolyes",
                "unchanged": "yes",
                "rootarray": [10, 20, 30],
            },
        ]
        return to_df(spark, data)

    @pytest.mark.parametrize(
        "field, function, expected",
        [
            pytest.param("id", lambda col: col + 1, [2, 3], id="increase id"),
            pytest.param(
                "firstname",
                lambda col: F.upper(col),
                ["CHRISTOPHER", "MIRIAM"],
                id="uppercase firstname",
            ),
            pytest.param(
                "lastname",
                lambda _: F.lit("foo"),
                ["foo"] * 2,
                id="column literal lastname",
            ),
        ],
    )
    def test_simple_function(
        self, spark: SparkSession, field: str, function, expected: list
    ):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add(field, function)

        result = runner.apply(df).orderBy("id").collect()

        assert [r[field] for r in result] == expected
        assert [r.unchanged for r in result] == ["yes", "yes"]

    def test_column(self, spark: SparkSession):
        """Test applying a column literal. This can't be done in a parameterized test
        as it errors when collecting if there isn't an active Spark session"""
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("id", F.lit("foo"))

        result = runner.apply(df)

        assert [r.id for r in result.orderBy("id").collect()] == ["foo", "foo"]

    def test_multiple_fields(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("id", lambda col: col + 1)
        runner.add("firstname", lambda col: F.upper(col))
        runner.add("lastname", F.lower)

        result = runner.apply(df).orderBy("id").collect()

        assert [r.id for r in result] == [2, 3]
        assert [r.firstname for r in result] == ["CHRISTOPHER", "MIRIAM"]
        assert [r.lastname for r in result] == ["lee", "margolyes"]
        assert [r.unchanged for r in result] == ["yes", "yes"]

    def test_create_new_root_field(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("newroot1", F.lit("yippee!"))
        runner.add("newroot2", F.concat("firstname", F.lit(" "), "lastname"))

        result = runner.apply(df).orderBy("id")

        # ensure new columns added after existing
        assert result.columns == df.columns + ["newroot1", "newroot2"]

        rows = result.select("newroot1", "newroot2").collect()
        assert rows[0] == ("yippee!", "Christopher Lee")
        assert rows[1] == ("yippee!", "Miriam Margolyes")

    def test_array(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("rootarray", lambda col: col + 1)

        result = runner.apply(df).orderBy("id").collect()

        assert [r.rootarray for r in result] == [[2, 3], [11, 21, 31]]


class TestNested:
    def _source_data(self, spark: SparkSession):
        data = [
            {
                "pets": {
                    "has_pets": "yes",
                    "pet_names": ["Fido", "Spot"],
                    "pet_details": [
                        {
                            "some": [
                                {"contrived": {"array": [{"setup": "   SILLINESS  "}]}}
                            ],
                        },
                    ],
                },
                "hoomans": [
                    {
                        "names": ["Jack", "Jill"],
                        "address": {
                            "house": 23,
                            "street": "A Nice Street",
                        },
                    }
                ],
            },
        ]
        return to_df(spark, data)

    def test_nested(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("pets.has_pets", F.upper)

        result = runner.apply(df).collect()

        assert [r.pets.has_pets for r in result] == ["YES"]

    def test_nested_array(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("pets.pet_names", F.upper)

        result = runner.apply(df).collect()

        assert result[0].pets.pet_names == ["FIDO", "SPOT"]

    def test_very_nested_array(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("pets.pet_details.some.contrived.array.setup", F.lower)
        runner.add("pets.pet_details.some.contrived.array.setup", F.trim)

        result = runner.apply(df).collect()

        assert (
            result[0].pets.pet_details[0].some[0].contrived.array[0].setup
            == "silliness"
        )

    def test_root_array(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add(
            "hoomans.names",
            lambda val: F.when(val == "Jack", F.upper(val)).otherwise(val),
        )

        result = runner.apply(df).collect()

        assert result[0].hoomans[0].names == ["JACK", "Jill"]

    def test_create_new_nested_field(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("newroot1", F.lit("yippee!"))
        runner.add("pets.pet_details.foo", F.lit("bar!"))
        runner.add("hoomans.address.city", F.lit("baz!"))

        result = runner.apply(df)

        assert len(result.columns) == len(df.columns) + 1

        rows = result.collect()
        for row in rows:
            assert row.newroot1 == "yippee!"
            assert row.pets.pet_details[0].foo == "bar!"
            assert row.hoomans[0].address.asDict() == {
                "house": 23,
                "street": "A Nice Street",
                "city": "baz!",
            }


class TestMechanics:
    def _source_data(self, spark: SparkSession):
        data = [
            {
                "id": 1,
                "nested": [
                    {
                        "schmested": {
                            "foo": 1,
                            "bar": 2,
                            "baz": [3],
                        },
                        "beep": "boop",
                    },
                ],
                "something": "else",
            },
        ]
        return to_df(spark, data)

    def test_apply_called_once_per_root_with_no_configs(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        with mock.patch.object(
            runner, runner._apply.__name__, wraps=runner._apply
        ) as mock_apply:
            _ = runner.apply(df)

        assert mock_apply.call_count == len(df.columns)

    @pytest.mark.parametrize(
        "field", ["nested.schmested.foo", "nested.schmested.baz", "nested.beep"]
    )
    def test_apply_called_once_per_member_in_path_to_leaf(
        self, spark: SparkSession, field: str
    ):
        df = self._source_data(spark).select("nested")

        runner = Functioniser()

        runner.add(field, F.lit("x"))

        with mock.patch.object(
            runner, runner._apply.__name__, wraps=runner._apply
        ) as mock_apply:
            _ = runner.apply(df)

        assert mock_apply.call_count == len(field.split("."))

    def test_configs_case_insensitive(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("iD", F.lit("x")).add("nested.schmested.BAZ", F.lit("y"))

        result = runner.apply(df)

        result = result.collect()

        assert result[0].id == "x"
        assert result[0].nested[0].schmested.baz == ["y"]

    @pytest.mark.parametrize("field", ["rooty", "nested.schmested.f00b4r"])
    def test_emits_warning_for_field_not_present(
        self, spark: SparkSession, caplog: pytest.LogCaptureFixture, field: str
    ):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("iD", F.lit("x")).add(field, F.lit("ain't there, buddy"))

        with caplog.at_level(logging.WARNING):
            result = runner.apply(df)

        assert field in caplog.text
        assert "iD" not in caplog.text

        result = result.collect()
        assert result[0].id == "x"

    def test_error_for_parent_struct_not_present(self, spark: SparkSession):
        df = self._source_data(spark)

        runner = Functioniser()

        runner.add("nested.schmested.a.b.c", F.lit("ain't there, buddy"))

        with pytest.raises(
            ValueError, match="Parent struct for nested.schmested.a.b.c not found"
        ):
            runner.apply(df)


class TestStringConfigs:
    def test_spark_builtins_in_internal(self, spark: SparkSession):
        """check a few builtins are found"""
        runner = Functioniser()

        assert "lower" in runner._spark_fns
        assert "upper" in runner._spark_fns
        assert "trim" in runner._spark_fns

        result = (
            to_df(spark, {"a": "aA", "b": "bB", "c": "  cC  "})
            .select(
                runner._spark_fns["lower"]("a"),  # type: ignore
                runner._spark_fns["upper"]("b"),  # type: ignore
                runner._spark_fns["trim"]("c"),  # type: ignore
            )
            .collect()
        )

        assert result[0] == ("aa", "BB", "cC")

    def test_spark_builtins_by_name(self, spark: SparkSession):
        """check a few builtins are usable by name"""
        runner = Functioniser().add("a", "lower").add("b", "upper").add("c", "trim")

        df = to_df(spark, {"a": "aA", "b": "bB", "c": "  cC  "})

        result = runner.apply(df)

        assert result.collect()[0] == ("aa", "BB", "cC")

    def test_custom_function(self, spark: SparkSession):
        runner = Functioniser()

        runner.register_function("foo", lambda _: F.lit("bar"))
        runner.add("a", "foo")

        result = runner.apply(to_df(spark, {"a": "a"}))

        assert result.collect()[0].a == "bar"

    def test_custom_function_override_builtin(self, spark: SparkSession):
        runner = Functioniser()

        runner.register_function("trim", lambda _: F.lit("TRIMMED"))
        runner.add("a", "trim")

        result = runner.apply(to_df(spark, {"a": "a"}))

        assert result.collect()[0].a == "TRIMMED"

    def test_not_found(self):
        runner = Functioniser()

        with pytest.raises(ValueError, match="foo"):
            runner.add("a", "foo")


class TestConfigDriven:
    def test_with_config(self, spark: SparkSession):
        df = to_df(spark, {"a": "aA", "b": "bB", "c": " cC "})

        config = {
            "a": "upper",
            "b": "lower",
            "c": "trim",
        }

        runner = Functioniser()
        for f, fn in config.items():
            runner.add(f, fn)

        result = runner.apply(df).collect()[0]

        assert result.a == "AA"
        assert result.b == "bb"
        assert result.c == "cC"


class TestReadme:
    """Sanity check code in the readme"""

    def test_setup(self, spark):
        df = to_df(
            spark,
            {
                "some_root": "a/b/c",
                "another_root": {
                    "nested_field": "baz",
                    "another_nested_field": "abbage",
                },
            },
        )

        runner = (
            Functioniser()
            .add("some_root", "upper")
            .add("another_root.nested_field", F.lit("foo"))
            .add(
                "another_root.another_nested_field",
                lambda col: F.concat(
                    F.split_part("some_root", F.lit("/"), F.lit(3)), col
                ),
            )
        )

        result = runner.apply(df).collect()

        assert result[0].asDict(True) == {
            "some_root": "A/B/C",
            "another_root": {
                "nested_field": "foo",
                "another_nested_field": "cabbage",
            },
        }

    def test_custom(self, spark: SparkSession):
        df = to_df(
            spark,
            {
                "my_root": {
                    "field_a": "  trim me  ",
                    "field_b": "123456",
                    "field_c": ["", "x"],
                }
            },
        )

        parsed_yaml = yaml.safe_load(
            """
            functions:
              my_root.field_a: trim
              my_root.field_b: substr_first_four
              my_root.field_c: set_to_foo
        """
        )

        runner = Functioniser()

        runner.register_function(
            "substr_first_four", lambda col: F.substring(col, 0, 4)
        )
        runner.register_function("set_to_foo", F.lit("foo"))

        for field, fn in parsed_yaml["functions"].items():
            runner.add(field, fn)

        result = runner.apply(df).collect()

        assert result[0].my_root == ("trim me", "1234", ["foo", "foo"])
