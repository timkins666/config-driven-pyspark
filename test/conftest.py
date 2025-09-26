import atexit
import json
import pytest
from pyspark.sql import SparkSession


@pytest.fixture(name="spark", scope="session")
def _spark():
    spark = SparkSession.builder.getOrCreate()
    atexit.register(spark.stop)
    yield spark


def to_df(spark, data: dict | list[dict]):
    return spark.read.json(
        spark.sparkContext.parallelize(
            [json.dumps(data if isinstance(data, list) else [data])]
        )
    )
