"""Harvesters for public external run tables.

Each source module documents exactly which public artifact it pulls, how the
``compute`` axis is derived, and which continuous metric its ``bpb`` column
carries (recorded per row in ``metric_name``). None of them download model
weights; they read released evaluation tables only.
"""
