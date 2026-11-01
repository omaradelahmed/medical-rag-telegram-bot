"""Diagnostic: does the expected page appear in search results, and at which rank?
Also prints the chunks stored for that page (start and end of each).
Run:  uv run debug_search.py
"""
import retriever

SS = "Study_Smarter_IM_Paper1.pdf"
TESTS = [
    ("complications of myocardial infarction", (SS, 9)),
    ("clinical features of chronic kidney disease", (SS, 63)),
    ("treatment of acute severe asthma", (SS, 80)),
]

for q, target in TESTS:
    print(f"\n=== {q}   (expected: {target[0]} p.{target[1]})")
    print("Variants:", retriever.expand_query(q))
    results = retriever.search(q, top_k=30)
    ranks = [i for i, r in enumerate(results, 1) if (r["source"], r["page"]) == target]
    print("Target page rank(s):", ranks or "not in top 30")
    for i, r in enumerate(results[:5], 1):
        mark = "  <== TARGET" if (r["source"], r["page"]) == target else ""
        print(f"{i:>2}. [{r['score']:.3f}] {r['source']} p.{r['page']}{mark}")

# Show how the first target page was chunked
src, page = TESTS[0][1]
print(f"\n=== Chunks stored for {src} p.{page}")
for n, r in enumerate((r for r in retriever._records if (r["source"], r["page"]) == (src, page)), 1):
    t = r["text"]
    print(f"[chunk {n}] START: {t[:170]}")
    print(f"           END:   ...{t[-200:]}\n")
