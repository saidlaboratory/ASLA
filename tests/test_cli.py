import pytest

from asla.cli import _crossover_root_message, build_parser


def test_demo_trials_must_be_positive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["demo", "--trials", "0"])


def test_crossover_root_message_does_not_fake_chinchilla_budget():
    msg = _crossover_root_message(None, "a", "b", (1.0, 2.0, 4.0), "chinchilla")
    assert "not reported" in msg
