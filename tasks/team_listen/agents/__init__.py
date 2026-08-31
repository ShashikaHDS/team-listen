"""Agent configs for the Team Listen tasks (M1_SPEC 1.14).

This package exists so the registry entry-point strings
``"tasks.team_listen.agents:skrl_mappo_cfg.yaml"`` resolve: Isaac Lab's
``load_cfg_from_registry`` imports the module left of the colon and joins
the yaml filename onto the module's directory.

Files:

* ``skrl_mappo_cfg.yaml`` -- PROVISIONAL until re-copied verbatim from the
  pinned checkout's cart_double_pendulum config on the 5090 (see the file's
  header and M1_SPEC section 0/1.14; enforced by tests/test_yaml_keyset.py).
* ``skrl_ippo_cfg.yaml`` -- the paired IPPO control config (spec 1.14 /
  8.2 step 6); produced by the same copy-then-edit procedure on the 5090.
"""
