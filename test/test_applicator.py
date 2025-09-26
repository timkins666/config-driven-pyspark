from pyspark.sql import SparkSession, functions as F
import pytest
from unittest import mock

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
                        "address": ["house", "street"],
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
            "hoomans.address",
            lambda val: F.when(val == "street", F.upper(val)).otherwise(val),
        )

        result = runner.apply(df).collect()

        assert result[0].hoomans[0].address == ["house", "STREET"]


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

        result = result.collect()
        assert result[0].id == "x"
