J_PER_KWH = 3.6e6


def joules_to_kwh(joules: float) -> float:
    return joules / J_PER_KWH


def energy_to_carbon_g(energy_kwh: float, intensity_gco2e_kwh: float) -> float:
    """Operational emissions in gCO2e: E[kWh] * grid carbon intensity [gCO2e/kWh]."""
    return energy_kwh * intensity_gco2e_kwh
