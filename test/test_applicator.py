from pyspark.sql import SparkSession, functions as F
import pytest

from applicator import DataframeFunctions
from test.conftest import to_df


class TestRootColumns:
    def _source_data(self, spark):
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
    def test_simple_function(self, spark, field: str, function, expected: list):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add(field, function)

        result = runner.run(df).orderBy("id").collect()

        assert [r[field] for r in result] == expected
        assert [r.unchanged for r in result] == ["yes", "yes"]

    def test_column(self, spark):
        """Test applying a column literal. This can't be done in a parameterized test
        as it errors when collecting if there isn't an active Spark session"""
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add("id", F.lit("foo"))

        result = runner.run(df)

        assert [r.id for r in result.orderBy("id").collect()] == ["foo", "foo"]

    def test_multiple_fields(self, spark):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add("id", lambda col: col + 1)
        runner.add("firstname", lambda col: F.upper(col))
        runner.add("lastname", F.lower)

        result = runner.run(df).orderBy("id").collect()

        assert [r.id for r in result] == [2, 3]
        assert [r.firstname for r in result] == ["CHRISTOPHER", "MIRIAM"]
        assert [r.lastname for r in result] == ["lee", "margolyes"]
        assert [r.unchanged for r in result] == ["yes", "yes"]

    def test_array(self, spark):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add("rootarray", lambda col: col + 1)

        result = runner.run(df).orderBy("id").collect()

        assert [r.rootarray for r in result] == [[2, 3], [11, 21, 31]]


class TestNested:
    def _source_data(self, spark):
        data = [
            {
                "pets": {
                    "has_pets": "yes",
                    "pet_names": ["Fido", "Spot"],
                    "pet_details": [
                        {
                            "name": "Fido",
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

    def test_nested(self, spark):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add("pets.has_pets", F.upper)

        result = runner.run(df).collect()

        assert [r.pets.has_pets for r in result] == ["YES"]

    def test_nested_array(self, spark):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add("pets.pet_names", F.upper)

        result = runner.run(df).collect()

        assert result[0].pets.pet_names == ["FIDO", "SPOT"]

    def test_very_nested_array(self, spark):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add("pets.pet_names.some.contrived.array.setup", F.lower)
        runner.add("pets.pet_names.some.contrived.array.setup", F.trim)

        result = runner.run(df).collect()

        assert result[0].pets.pet_names.some[0].contrived.array[0].setup == [
            "silliness"
        ]

    def test_root_array(self, spark):
        df = self._source_data(spark)

        runner = DataframeFunctions()

        runner.add(
            "hoomans.address",
            lambda val: F.when(val == "street", F.upper(val)).otherwise(val),
        )

        result = runner.run(df).collect()

        assert result[0].hoomans[0].address == ["house", "STREET"]
