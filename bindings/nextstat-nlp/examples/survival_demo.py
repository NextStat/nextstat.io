from __future__ import annotations

from nextstat_nlp import extract_survival_records


def main() -> None:
    texts = [
        "Subject 001: progressed at day 84. Age 63. Dose 20 mg daily.",
        "Subject 002: withdrew at week 12 (lost to follow-up). Age 58. Dose 10 mg daily.",
    ]
    ds = extract_survival_records(texts, backend="heuristic")
    time, event, X, feature_names = ds.to_design_matrix()
    print("time", time)
    print("event", event)
    print("X", X)
    print("features", feature_names)


if __name__ == "__main__":
    main()

