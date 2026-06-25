import pytest

from asla.cli import build_parser


def test_demo_trials_must_be_positive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--trials", "0"])
