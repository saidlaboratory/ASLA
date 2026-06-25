from asla.cli import _runtime_config, build_parser
from asla.config import AuditConfig


def test_default_counts_are_paper_grade():
    cfg = AuditConfig()
    assert cfg.counts.n_boot == 1000
    assert cfg.counts.n_trials == 500


def test_fast_flag_uses_small_counts():
    parser = build_parser()
    args = parser.parse_args(["demo", "--fast"])
    _, n_boot, n_trials = _runtime_config(args)
    cfg = AuditConfig()
    assert n_boot == cfg.counts.fast_n_boot
    assert n_trials == cfg.counts.fast_n_trials

