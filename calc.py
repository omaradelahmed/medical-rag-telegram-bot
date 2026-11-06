"""Medical calculators: deterministic formula-based math, exposed as MCP tools.
Why a separate module: the LLM never computes these numbers - the code does,
so every figure in the bot's answer is exact. Add a calculator here and a thin
@mcp.tool wrapper in medical_server.py.

Formulas / sources:
  eGFR           CKD-EPI 2021 (creatinine in mg/dL)
  body area      Mosteller
  BMI            WHO classification
  CHA2DS2-VASc   AHA/ACC/HRS guideline, AF anticoagulation
  CRB-65         CRB-65 / CURB-65, community-acquired pneumonia
"""

def _bad(msg: str) -> str:
    """Error format the LLM understands and can pass on."""
    return f"ERROR: {msg}"


def bmi(weight_kg: float, height_cm: float) -> str:
    if not weight_kg > 0 or not height_cm >= 100 or height_cm > 250:
        return _bad("need a positive weight_kg and a height_cm between 100 and 250")
    v = weight_kg / (height_cm / 100) ** 2
    cat = ("underweight (<18.5)" if v < 18.5
           else "normal (18.5-24.9)" if v < 25
           else "overweight (25-29.9)" if v < 30
           else "obesity (>=30)")
    return f"BMI = {v:.1f} kg/m2 -> {cat} (WHO classification)"


def body_surface_area(weight_kg: float, height_cm: float) -> str:
    if not weight_kg > 0 or not height_cm >= 100 or height_cm > 250:
        return _bad("need a positive weight_kg and a height_cm between 100 and 250")
    v = 0.007189 * (weight_kg ** 0.42) * (height_cm ** 0.718)
    return f"Body surface area = {v:.2f} m2 (Mosteller formula)"


def calc_gfr(creatinine_mg_dl: float, age: int, sex: str) -> str:
    """CKD-EPI 2021, creatinine-based (kdigo 2013 recommendation)."""
    s = (sex or "").strip().lower()
    if not s in ("m", "male", "f", "female"):
        return _bad("sex must be 'male' or 'female'")
    female = s in ("f", "female")
    if not creatinine_mg_dl > 0:
        return _bad("creatinine_mg_dl must be a positive number (mg/dL)")
    if not 18 <= age <= 110:
        return _bad("age must be between 18 and 110 years")
    k = 0.7 if female else 0.9
    alpha = -0.302 if female else -0.411
    r = creatinine_mg_dl / k
    v = 142 * min(r, 1) ** alpha * max(r, 1) ** -1.200 * 0.9938 ** age * (1.012 if female else 1.0)
    cat = (">=90  (G1, normal/high)" if v >= 90
           else "60-89 (G2, mildly decreased)" if v >= 60
           else "45-59 (G3a, mildly-moderately decreased)" if v >= 45
           else "30-44 (G3b, moderately-severely decreased)" if v >= 30
           else "15-29 (G4, severely decreased)" if v >= 15
           else "<15 (G5, kidney failure)")
    return f"eGFR = {v:.0f} mL/min/1.73m2  (CKD-EPI 2021) -> {cat}"


def chadsvat_score(age: int, hypertension: bool, diabetes: bool, heart_failure: bool,
                   stroke_tia_or_thromboembolism: bool, prior_mh_or_atherosclerosis: bool,
                   female: bool = False) -> str:
    """CHA2DS2-VASc for stroke risk in atrial fibrillation (includes the sex-category point)."""
    if not 18 <= age <= 110:
        return _bad("age must be between 18 and 110 years")
    s, parts = 0, []
    if age >= 75:
        s += 2; parts.append("age >=75 (2 points)")
    elif age >= 65:  # age scores are mutually exclusive: 0 / 1 (65-74) / 2 (>=75)
        s += 1; parts.append("age 65-74 (1 point)")
    for label, flag in (
        ("prior stroke/TIA/thromboembolism (2 points)", stroke_tia_or_thromboembolism),
        ("hypertension (1 point)", hypertension),
        ("diabetes (1 point)", diabetes),
        ("heart failure or LV dysfunction (1 point)", heart_failure),
        ("vascular disease: prior MI/atherosclerosis/aortic plaque (1 point)",
         prior_mh_or_atherosclerosis),
        ("sex category: female (1 point)", female),
    ):
        if flag:
            s += 2 if " (2 points)" in label else 1
            parts.append(label)
    if not parts:
        parts = ["no risk factors present (0 points)"]
    rec = ("low risk, no antithrombotic therapy needed"
           if s == 0
           else "score 1: consider oral anticoagulation (individual decision)"
           if s == 1
           else "score >=2: oral anticoagulation recommended")
    return f"CHA2DS2-VASc = {s}  [ " + " + ".join(parts) + f" ]\nRecommendation: {rec}"


def crb65(confusion: bool, resp_rate_per_min: int,
          sbp_mmhg: int, dbp_mmhg: int, age: int) -> str:
    """CRB-65 for community-acquired pneumonia severity (CURB-65 without urea)."""
    if resp_rate_per_min < 0 or sbp_mmhg < 50 or dbp_mmhg < 30 or not 1 <= age <= 120:
        return _bad("check the values: resp_rate>=0, sbp/dbp physiological, age 1-120")
    s, parts = 0, []
    if confusion:
        s += 1; parts.append("confusion (1)")
    if resp_rate_per_min > 30:
        s += 1; parts.append(f"respiratory rate {resp_rate_per_min}/min >30 (1)")
    if sbp_mmhg < 90 or dbp_mmhg <= 60:
        s += 1; parts.append(f"BP {sbp_mmhg}/{dbp_mmhg} (SBP<90 or DBP<=60) (1)")
    if age > 65:
        s += 1; parts.append(f"age {age} >65 (1)")
    if not parts:
        parts = ["no criteria met (0)"]
    sev = ("0: low risk, home treatment candidate"
           if s == 0
           else "1: low risk, home treatment with close follow-up"
           if s == 1
           else "score 2: consider hospital admission"
           if s == 2
           else "score 3: moderate-high mortality, hospital admission"
           if s == 3
           else "score 4: high mortality (CURB-65 includes urea >7 mmol/L), consider ICU")
    return f"CRB-65 = {s}  [ " + " + ".join(parts) + f" ]\nSeverity: {sev}"
