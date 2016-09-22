from collections import OrderedDict
import json

import pytest

from eemeter.io.serializers.meter_output import serialize_derivative_pairs
from eemeter.ee.derivatives import DerivativePair, Derivative


@pytest.fixture
def derivative_pairs():
    return [
        DerivativePair(
            "interpretation1",
            Derivative("1", 10, 3, 3, 5,
                       OrderedDict([('2011-01-01T00:00:00+00:00', 32.0)])),
            Derivative("2", 10, 6, 6, 5, None),
        ),
        DerivativePair(
            "interpretation2",
            None,
            Derivative("2", 10, 8, 8, 5, None),
        ),
    ]


def test_basic_usage(derivative_pairs):
    serialized = serialize_derivative_pairs(derivative_pairs)
    assert len(serialized) == 2
    assert len(json.dumps(serialized)) == 420
    assert serialized[0]["interpretation"] == "interpretation1"
    assert serialized[0]["baseline"]["label"] == "1"
    assert serialized[0]["baseline"]["value"] == 10
    assert serialized[0]["baseline"]["lower"] == 3
    assert serialized[0]["baseline"]["upper"] == 3
    assert serialized[0]["baseline"]["n"] == 5
    assert serialized[0]["baseline"]["demand_fixture"] is not None
    assert serialized[0]["reporting"]["value"] == 10
    assert serialized[0]["reporting"]["demand_fixture"] is None

    assert serialized[1]["interpretation"] == "interpretation2"
    assert serialized[1]["baseline"] is None
    assert serialized[1]["reporting"]["value"] == 10
