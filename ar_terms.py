"""Arabic medical glossary: builds the English search query WITHOUT the LLM.
Why: the small LLM mistranslated "احتشاء" (infarction) as "stripping".
Extend DISEASES / QUESTION_TYPES whenever a question is missed (it then falls back to the LLM).
"""
import re

_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")


def _norm(text: str) -> str:
    """Normalize Arabic spelling so matching is forgiving."""
    text = _DIACRITICS.sub("", text)
    text = re.sub("[أإآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"(?<!\w)ال(?=\w)", "", text)  # drop the definite article "ال"
    return text.lower()


# What the user wants to know  ->  the books' wording
QUESTION_TYPES = {
    "مضاعفات": "complications of",
    "اسباب": "causes of", "أسباب": "causes of", "سبب": "causes of",
    "اعراض": "clinical picture of", "أعراض": "clinical picture of",
    "علامات": "clinical picture of", "الصورة السريرية": "clinical picture of",
    "علاج": "treatment of", "العلاج": "treatment of", "ازاي يتعالج": "treatment of",
    "تشخيص": "diagnosis of", "فحوصات": "investigations of", "تحاليل": "investigations of",
    "تعريف": "definition of",
    "عوامل الخطر": "risk factors of", "عوامل الخطورة": "risk factors of",
    "تدبير": "management of", "ادارة": "management of", "إدارة": "management of",
    "انواع": "types of", "أنواع": "types of",
    "ادوية": "drugs used in", "أدوية": "drugs used in",
}

# Arabic term -> English term (longest match wins, so specific terms go first automatically)
DISEASES = {
    # cardiology
    "احتشاء عضلة القلب": "myocardial infarction", "احتشاء القلب": "myocardial infarction",
    "جلطة القلب": "myocardial infarction", "الجلطة القلبية": "myocardial infarction",
    "النوبة القلبية": "myocardial infarction",
    "الذبحة الصدرية": "angina", "الذبحة غير المستقرة": "unstable angina",
    "الفشل القلبي": "heart failure", "قصور القلب": "heart failure", "فشل القلب": "heart failure",
    "ارتفاع ضغط الدم": "hypertension", "ضغط الدم المرتفع": "hypertension", "فرط ضغط الدم": "hypertension",
    "الرجفان الاذيني": "atrial fibrillation", "الرجفان الأذيني": "atrial fibrillation",
    "التهاب الشغاف": "infective endocarditis", "التهاب بطانة القلب": "infective endocarditis",
    "الحمى الروماتيزمية": "rheumatic fever", "اعتلال عضلة القلب": "cardiomyopathy",
    "التهاب التامور": "pericarditis", "الاغماء": "syncope", "الإغماء": "syncope",
    # hematology
    "فقر الدم": "anemia", "فقر الدم الانحلالي": "hemolytic anemia",
    "فقر الدم اللاتنسجي": "aplastic anemia", "نقص الحديد": "iron deficiency anemia",
    "الثلاسيميا": "thalassemia", "فقر الدم المنجلي": "sickle cell disease",
    "نقص الصفائح": "thrombocytopenia", "ابيضاض الدم": "leukemia", "سرطان الدم": "leukemia",
    "الليمفوما": "lymphoma", "نقل الدم": "blood transfusion", "ضخامة الطحال": "splenomegaly",
    # liver / GI
    "تليف الكبد": "liver cirrhosis", "تشمع الكبد": "liver cirrhosis",
    "الاستسقاء": "ascites", "الاستسقاء البطني": "ascites",
    "الغيبوبة الكبدية": "hepatic encephalopathy", "اعتلال الدماغ الكبدي": "hepatic encephalopathy",
    "الفشل الكبدي": "liver failure", "اليرقان": "jaundice", "الصفراء": "jaundice",
    "التهاب الكبد الفيروسي": "viral hepatitis", "التهاب الكبد ب": "hepatitis B",
    "التهاب الكبد سي": "hepatitis C", "نزيف الجهاز الهضمي": "gastrointestinal bleeding",
    "القيء الدموي": "hematemesis", "قرحة المعدة": "peptic ulcer", "القرحة الهضمية": "peptic ulcer",
    "ارتجاع المريء": "GERD", "التهاب البنكرياس": "pancreatitis",
    "القولون العصبي": "irritable bowel syndrome", "التهاب القولون التقرحي": "ulcerative colitis",
    "مرض كرون": "Crohn's disease", "الاسهال": "diarrhea", "الإسهال": "diarrhea",
    "دوالي المريء": "esophageal varices", "مرض السيلياك": "celiac disease",
    # kidney
    "الفشل الكلوي المزمن": "chronic kidney disease", "القصور الكلوي المزمن": "chronic kidney disease",
    "الفشل الكلوي الحاد": "acute kidney injury", "الاصابة الكلوية الحادة": "acute kidney injury",
    "الفشل الكلوي": "renal failure", "المتلازمة الكلوية": "nephrotic syndrome",
    "التهاب كبيبات الكلى": "glomerulonephritis", "غسيل الكلى": "hemodialysis",
    "فرط البوتاسيوم": "hyperkalemia", "ارتفاع البوتاسيوم": "hyperkalemia",
    "نقص البوتاسيوم": "hypokalemia", "نقص الصوديوم": "hyponatremia",
    "الحماض الاستقلابي": "metabolic acidosis", "التهاب الحويضة والكلية": "pyelonephritis",
    "البيلة الدموية": "hematuria", "البروتين في البول": "proteinuria",
    # chest
    "الربو": "bronchial asthma", "الربو القصبي": "bronchial asthma",
    "الانسداد الرئوي المزمن": "COPD", "مرض الانسداد الرئوي المزمن": "COPD",
    "الالتهاب الرئوي": "pneumonia", "ذات الرئة": "pneumonia",
    "السل": "tuberculosis", "الدرن": "tuberculosis",
    "الانصباب الجنبي": "pleural effusion", "الجلطة الرئوية": "pulmonary embolism",
    "الصمة الرئوية": "pulmonary embolism", "توسع القصبات": "bronchiectasis",
    "التليف الرئوي": "pulmonary fibrosis", "نفث الدم": "hemoptysis", "ضيق التنفس": "dyspnea",
    # rheumatology
    "الروماتويد": "rheumatoid arthritis", "التهاب المفاصل الروماتويدي": "rheumatoid arthritis",
    "الذئبة الحمراء": "SLE", "النقرس": "gout", "هشاشة العظام": "osteoporosis",
    "خشونة المفاصل": "osteoarthritis", "التصلب الجهازي": "systemic sclerosis",
    "التهاب الفقار اللاصق": "ankylosing spondylitis", "ظاهرة رينو": "Raynaud's phenomenon",
    "لين العظام": "osteomalacia",
    # neurology
    "السكتة الدماغية": "stroke", "الجلطة الدماغية": "stroke", "الصرع": "epilepsy",
    "الحالة الصرعية": "status epilepticus", "مرض باركنسون": "Parkinson's disease",
    "الشلل الرعاش": "Parkinson's disease", "التصلب المتعدد": "multiple sclerosis",
    "الوهن العضلي": "myasthenia gravis", "متلازمة غيلان باريه": "Guillain-Barre syndrome",
    "التهاب السحايا": "meningitis", "الصداع": "headache", "الصداع النصفي": "migraine",
    "الدوار": "vertigo", "الغيبوبة": "coma", "التهاب الاعصاب": "peripheral neuropathy",
    "الاعتلال العصبي المحيطي": "peripheral neuropathy", "شلل بيل": "Bell's palsy",
    "الشلل النصفي": "hemiplegia", "الشلل السفلي": "paraplegia", "الرنح": "ataxia",
    # endocrine
    "السكري": "diabetes mellitus", "داء السكري": "diabetes mellitus", "مرض السكر": "diabetes mellitus",
    "الحماض الكيتوني": "diabetic ketoacidosis", "نقص سكر الدم": "hypoglycemia", "هبوط السكر": "hypoglycemia",
    "فرط نشاط الغدة الدرقية": "hyperthyroidism", "التسمم الدرقي": "hyperthyroidism",
    "قصور الغدة الدرقية": "hypothyroidism", "خمول الغدة الدرقية": "hypothyroidism",
    "العاصفة الدرقية": "thyroid storm", "متلازمة كوشينغ": "Cushing's syndrome",
    "قصور الغدة الكظرية": "adrenal insufficiency", "مرض اديسون": "adrenal insufficiency",
    "ضخامة الاطراف": "acromegaly", "السمنة": "obesity", "المتلازمة الايضية": "metabolic syndrome",
    "فرط كالسيوم الدم": "hypercalcemia", "فرط نشاط الغدة جار الدرقية": "hyperparathyroidism",
    "تكيس المبايض": "polycystic ovary syndrome",
    # infections
    "الحمى مجهولة السبب": "fever of unknown origin", "الصدمة الانتانية": "septic shock",
    "الزحار": "dysentery", "الحمى": "fever",
}

_QTYPES = sorted(((_norm(k), v) for k, v in QUESTION_TYPES.items()), key=lambda kv: -len(kv[0]))
_DISEASES = sorted(((_norm(k), v) for k, v in DISEASES.items()), key=lambda kv: -len(kv[0]))


def _find(key: str, text: str):
    # optional one-letter Arabic prefix (و ب ل ف ك), key must start/end on a word boundary
    return re.search(rf"(?<!\w)[وبلفك]?{re.escape(key)}(?!\w)", text)


def arabic_to_query(question: str) -> str | None:
    """English search query built from the glossary, or None if no disease term is known."""
    text = _norm(question)
    diseases = []
    for key, eng in _DISEASES:
        m = _find(key, text)
        if m:
            if eng not in diseases:
                diseases.append(eng)
            text = text[:m.start()] + " " + text[m.end():]  # remove it so shorter keys don't re-match
    if not diseases:
        return None
    qtype = ""
    for key, eng in _QTYPES:
        if _find(key, text):
            qtype = eng
            break
    return f"{qtype} {' and '.join(diseases)}".strip()


if __name__ == "__main__":
    tests = [
        "ما هي مضاعفات احتشاء عضلة القلب؟",
        "ما هي أسباب الفشل القلبي؟",
        "ما هي أعراض الفشل الكلوي المزمن؟",
        "كيف يتم علاج الربو الحاد؟",
        "ما هو علاج العاصفة الدرقية",
        "ما هي عاصمة فرنسا؟",
        "ما هي مضاعفات السل",
    ]
    for t in tests:
        print(f"{t}  ->  {arabic_to_query(t)!r}")
