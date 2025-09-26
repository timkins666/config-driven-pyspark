from test.conftest import to_df
from utils import flatten_schema


def test_flatten_schema(spark):
    df = to_df(
        spark,
        [
            {
                "id": 1,
                "name": "John",
                "address": {
                    "street": "123 Main St",
                    "city": "New York",
                },
                "pets": [
                    {
                        "name": "dog",
                        "face": {
                            "eyes": [
                                {"which": "left", "colour": "brown"},
                                {"which": "right", "colour": "orange"},
                            ],
                            "nose": "yes",
                        },
                    }
                ],
            },
        ],
    )

    result = flatten_schema(df)

    assert result == [
        "address.city",
        "address.street",
        "id",
        "name",
        "pets[].face.eyes[].colour",
        "pets[].face.eyes[].which",
        "pets[].face.nose",
        "pets[].name",
    ]

