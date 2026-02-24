"""Unit tests for deterministic text parsers."""
from __future__ import annotations

import unittest

from nextstat_nlp._parsers import normalize_units, parse_event, parse_numeric, parse_time


class TestParseTime(unittest.TestCase):
    def test_days(self):
        self.assertEqual(parse_time("84 days"), 84.0)
        self.assertEqual(parse_time("survived 84 days"), 84.0)
        self.assertEqual(parse_time("at 1 day"), 1.0)

    def test_day_prefix(self):
        self.assertEqual(parse_time("day 84"), 84.0)
        self.assertEqual(parse_time("at day 10"), 10.0)

    def test_weeks(self):
        self.assertEqual(parse_time("12 weeks"), 84.0)
        self.assertEqual(parse_time("week 12"), 84.0)
        self.assertEqual(parse_time("at week 4"), 28.0)

    def test_months(self):
        self.assertAlmostEqual(parse_time("6 months"), 182.625, places=3)
        self.assertAlmostEqual(parse_time("month 6"), 182.625, places=3)
        self.assertAlmostEqual(parse_time("3.5 months"), 3.5 * 30.4375, places=3)

    def test_years(self):
        self.assertAlmostEqual(parse_time("2 years"), 730.5, places=3)
        self.assertAlmostEqual(parse_time("year 2"), 730.5, places=3)
        self.assertAlmostEqual(parse_time("1 year"), 365.25, places=3)

    def test_decimal(self):
        self.assertAlmostEqual(parse_time("2.5 days"), 2.5, places=3)
        self.assertAlmostEqual(parse_time("1.5 weeks"), 10.5, places=3)

    def test_no_match(self):
        self.assertIsNone(parse_time("no time info here"))
        self.assertIsNone(parse_time(""))

    def test_hours(self):
        self.assertAlmostEqual(parse_time("48 hours"), 2.0, places=3)


class TestParseEvent(unittest.TestCase):
    def test_event_keywords(self):
        self.assertEqual(parse_event("patient progressed"), 1)
        self.assertEqual(parse_event("patient died"), 1)
        self.assertEqual(parse_event("death occurred"), 1)
        self.assertEqual(parse_event("relapsed at day 10"), 1)
        self.assertEqual(parse_event("recurrence observed"), 1)
        self.assertEqual(parse_event("disease progression"), 1)

    def test_censored_keywords(self):
        self.assertEqual(parse_event("patient was censored"), 0)
        self.assertEqual(parse_event("withdrew from study"), 0)
        self.assertEqual(parse_event("lost to follow-up"), 0)
        self.assertEqual(parse_event("patient alive"), 0)
        self.assertEqual(parse_event("disease stable"), 0)
        self.assertEqual(parse_event("discontinued treatment"), 0)
        self.assertEqual(parse_event("completed the study"), 0)

    def test_no_match(self):
        self.assertIsNone(parse_event("received treatment"))
        self.assertIsNone(parse_event("enrolled in trial"))

    def test_ambiguous_conservative(self):
        # Both event + censored keywords -> conservative (censored)
        self.assertEqual(parse_event("progressed then withdrew"), 0)


class TestParseNumeric(unittest.TestCase):
    def test_integer(self):
        self.assertEqual(parse_numeric("63 years", "age"), 63.0)

    def test_float(self):
        self.assertEqual(parse_numeric("5.5 mg/kg", "dose"), 5.5)

    def test_dose(self):
        self.assertEqual(parse_numeric("20 mg", "dose"), 20.0)

    def test_no_number(self):
        self.assertIsNone(parse_numeric("no numbers", "age"))


class TestNormalizeUnits(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(normalize_units(10.0, "mg", "mg"), 10.0)

    def test_mg_to_g(self):
        self.assertAlmostEqual(normalize_units(1000.0, "mg", "g"), 1.0)

    def test_weeks_to_days(self):
        self.assertAlmostEqual(normalize_units(2.0, "weeks", "days"), 14.0)

    def test_months_to_days(self):
        self.assertAlmostEqual(normalize_units(1.0, "months", "days"), 30.4375)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            normalize_units(1.0, "apples", "oranges")


if __name__ == "__main__":
    unittest.main()
