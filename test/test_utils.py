from test.conftest import to_df
from utils import flatten_schema, limit_depth


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


class TestLimitDepth:
    def test_limit_depth(self):
        assert limit_depth("a.b.c.d", 2) == "a.b"
        assert limit_depth("a.b.c.d", 1) == "a"
        assert limit_depth("a.b.c.d", 0) == ""
        assert limit_depth("a.b.c.d", -1) == "a.b.c"

    def test_limit_depth_with_string(self):
        assert limit_depth("a.b.c.d", "c.d") == "a.b"
        assert limit_depth("a.b.c.d", "x") == "a"
        assert limit_depth("a.b.c.d", "1.2.3") == "a.b.c"

    def test_default(self):
        assert limit_depth("a.b.c.d") == "a.b.c"
