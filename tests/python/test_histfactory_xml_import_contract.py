from __future__ import annotations

import json
from pathlib import Path


def test_from_histfactory_xml_builds_model_and_matches_workspace_n_params():
    import nextstat

    xml_path = Path(__file__).resolve().parents[1] / "fixtures" / "histfactory" / "combination.xml"
    ws_path = Path(__file__).resolve().parents[1] / "fixtures" / "histfactory" / "workspace.json"

    m_xml = nextstat.from_histfactory_xml(xml_path)
    assert int(m_xml.n_params()) > 0

    ws = json.loads(ws_path.read_text(encoding="utf-8"))
    m_ws = nextstat.HistFactoryModel.from_workspace(json.dumps(ws))
    assert int(m_xml.n_params()) == int(m_ws.n_params())

    init = m_xml.suggested_init()
    nll = float(m_xml.nll(init))
    assert nll == nll  # NaN guard

