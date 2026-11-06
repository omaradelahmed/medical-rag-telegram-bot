"""Quick unit test for the medical calculators (formulas, run without any LLM).
Run:  uv run test_calc.py
"""
import calc

tests = [
    ("bmi", calc.bmi(80, 175)),
    ("bmi child-range", calc.bmi(40, 120)),
    ("bmi bad", calc.bmi(-5, 175)),
    ("bsa", calc.body_surface_area(80, 175)),
    ("gfr female", calc.calc_gfr(1.0, 65, "female")),
    ("gfr male high cr", calc.calc_gfr(2.6, 55, "male")),
    ("gfr bad sex", calc.calc_gfr(1.0, 55, "x")),
    ("chadsvat 0", calc.chadsvat_score(60, False, False, False, False, False)),
    ("chadsvat high", calc.chadsvat_score(80, True, True, True, True, False)),
    ("chadsvat female", calc.chadsvat_score(30, False, False, False, False, False, female=True)),
    ("crb65", calc.crb65(False, 24, 110, 70, 60)),
    ("crb65 severe", calc.crb65(True, 34, 82, 55, 78)),
]
for name, out in tests:
    print(f"--- {name}\n{out}\n")
print("done")
