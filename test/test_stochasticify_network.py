import pytest
import pypsa
import pandas as pd

from scripts.stochasticify_network import (
    apply_stochastic_config,
    _load_annual_energy_twh,
)


def _network():
    n = pypsa.Network()
    snapshots = pd.date_range("2020-01-01", periods=2, freq="h")
    n.set_snapshots(snapshots)
    n.snapshot_weightings.loc[:, "generators"] = 500_000.0
    n.add("Bus", "DE")
    n.add("Bus", "DE heat")
    n.add("Bus", "DE hp")
    n.add("Carrier", "gas boiler")
    loads = {
        "DE land transport oil": ("land transport oil", [1000.0, 1000.0]),
        "DE land transport EV": ("land transport EV", [100.0, 100.0]),
        "DE gas for industry": ("gas for industry", [1000.0, 1000.0]),
        "DE H2 for industry": ("H2 for industry", [100.0, 100.0]),
        "DE rural heat": ("rural heat", [1000.0, 1000.0]),
        "DE electricity": ("electricity", [100.0, 100.0]),
    }
    for name, (carrier, values) in loads.items():
        n.add("Load", name, bus="DE", carrier=carrier)
        n.loads_t.p_set.loc[:, name] = values
    n.add("Link", "DE rural air heat pump", bus0="DE", bus1="DE heat", carrier="rural air heat pump")
    n.links_t.efficiency.loc[:, "DE rural air heat pump"] = [2.0, 2.0]
    n.add("Link", "DE gas boiler", bus0="DE", bus1="DE heat", carrier="gas boiler", marginal_cost=10.0)
    return n


def _catalogue(enable=False, active_scenario=None):
    cfg = {
        "enable": enable,
        "settings": {
            "weighting": "generators",
            "allow_unmet_target": False,
            "tolerance_twh": 1e-6,
            "non_negative_tolerance": 1e-8,
        },
        "families": {
            "ELEC": {
                "targets": {"small": 100.0},
                "entries": {
                    "land_transport_oil": {
                        "source": "land transport oil",
                        "target": "land transport EV",
                        "cap": 0.2,
                        "type": "electrify_transport",
                        "source_efficiency": 16.0712,
                        "target_efficiency": 53.19,
                    }
                },
            },
            "OILGAS": {
                "targets": {"small": 120.0},
                "entries": {
                    "land_transport_oil": {
                        "source": "land transport oil",
                        "target": "land transport EV",
                        "cap": 1.0,
                        "type": "reverse_shift",
                        "source_efficiency": 1.0,
                        "target_efficiency": 1.0,
                        "max_target_reduction_fraction": 0.5,
                        "strict_target_available": True,
                    },
                    "gas_for_industry": {
                        "source": "gas for industry",
                        "target": "H2 for industry",
                        "cap": 1.0,
                        "type": "reverse_shift",
                        "source_efficiency": 1.0,
                        "target_efficiency": 1.0,
                        "max_target_reduction_fraction": 1.0,
                        "strict_target_available": True,
                    },
                },
            },
        },
        "scenario_definitions": {
            "BASE": "base",
            "ELEC_HEAT": {
                "demand_transition": {
                    "family": "ELEC",
                    "target": "small",
                    "priority": ["land_transport_oil"],
                }
            },
            "OILGAS_ROAD": {
                "demand_transition": {
                    "family": "OILGAS",
                    "target": "small",
                    "priority": ["land_transport_oil", "gas_for_industry"],
                }
            },
            "PRICE_GAS_HIGH": {
                "modify_components": [
                    {
                        "component": "Link",
                        "attribute": "marginal_cost",
                        "carrier": "gas boiler",
                        "operation": "scale",
                        "value": 1.2,
                    }
                ]
            },
        },
    }
    if active_scenario is not None:
        cfg["active_scenario"] = active_scenario
    if enable:
        cfg["scenarios"] = {"BASE": 0.5, "ELEC_HEAT": 0.5}
    return cfg


def test_deterministic_active_scenario_modifies_loads_without_set_scenarios(monkeypatch):
    n = _network()
    called = False

    def fail_set_scenarios(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("set_scenarios must not be called")

    monkeypatch.setattr(n, "set_scenarios", fail_set_scenarios)
    apply_stochastic_config(n, {}, _catalogue(enable=False, active_scenario="ELEC_HEAT"))

    assert not called
    assert _load_annual_energy_twh(n, "DE land transport oil") == pytest.approx(900.0)
    assert _load_annual_energy_twh(n, "DE land transport EV") > 100.0


def test_stochastic_catalogue_applies_only_corresponding_scenario():
    n = _network()
    apply_stochastic_config(n, {}, _catalogue(enable=True))

    assert isinstance(n.loads_t.p_set.columns, pd.MultiIndex)
    base_oil = _load_annual_energy_twh(n, "DE land transport oil", scenario="BASE")
    elec_oil = _load_annual_energy_twh(n, "DE land transport oil", scenario="ELEC_HEAT")
    assert base_oil == pytest.approx(1000.0)
    assert elec_oil == pytest.approx(900.0)


def test_reverse_shift_caps_target_and_continues_priority():
    n = _network()
    apply_stochastic_config(n, {}, _catalogue(enable=False, active_scenario="OILGAS_ROAD"))

    assert _load_annual_energy_twh(n, "DE land transport EV") == pytest.approx(50.0)
    assert _load_annual_energy_twh(n, "DE H2 for industry") == pytest.approx(30.0)
    assert _load_annual_energy_twh(n, "DE land transport oil") == pytest.approx(1050.0)
    assert _load_annual_energy_twh(n, "DE gas for industry") == pytest.approx(1070.0)
    assert (n.loads_t.p_set >= -1e-8).all().all()


def test_modify_components_still_works():
    n = _network()
    apply_stochastic_config(n, {}, _catalogue(enable=False, active_scenario="PRICE_GAS_HIGH"))
    assert n.links.loc["DE gas boiler", "marginal_cost"] == pytest.approx(12.0)



def test_definitions_alias_is_accepted():
    n = _network()
    cfg = _catalogue(enable=False, active_scenario="PRICE_GAS_HIGH")
    cfg["definitions"] = cfg.pop("scenario_definitions")

    apply_stochastic_config(n, {}, cfg)

    assert n.links.loc["DE gas boiler", "marginal_cost"] == pytest.approx(12.0)

def test_missing_scenario_definition_raises_clear_error():
    n = _network()
    cfg = _catalogue(enable=True)
    cfg["scenarios"] = {"BASE": 0.5, "MISSING": 0.5}
    with pytest.raises(KeyError, match="Missing scenario_definitions"):
        apply_stochastic_config(n, {}, cfg)
