from __future__ import annotations

from nextstat_nlp.regimens import _canonicalize_units, _guess_infusion_duration_from_text, _guess_freq_from_text, _guess_route_from_text


def test_guess_route_iv_infusion() -> None:
    txt = "Administer 1000 mg intravenous over 2 hours."
    assert _guess_route_from_text(txt) == "IV"


def test_guess_route_oral() -> None:
    txt = "Take 400 mg orally once daily."
    assert _guess_route_from_text(txt) == "oral"


def test_guess_freq_q12h() -> None:
    txt = "Dose: 250 mg every 12 hours."
    assert _guess_freq_from_text(txt) == "Q12H"


def test_guess_freq_bid() -> None:
    txt = "30 mg daily in 2 to 3 divided doses; twice daily dosing is allowed."
    assert _guess_freq_from_text(txt) == "BID"


def test_guess_duration_over_hours() -> None:
    txt = "1000 mg intravenous over 2 hours"
    dur = _guess_infusion_duration_from_text(txt)
    assert dur is not None
    # parse_time returns days; 2 hours = 2/24 days
    assert abs(dur - (2.0 / 24.0)) < 1e-6


def test_canonicalize_units_filters_strength_units() -> None:
    assert _canonicalize_units("mcg/mL") == ""
    assert _canonicalize_units("mg/kg") == "mg/kg"
    assert _canonicalize_units("μg") == "mcg"
