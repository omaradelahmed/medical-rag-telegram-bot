# Demo questions

Try these after copying `examples/demo_docs/demo_reference.pdf` into `docs/` and running `uv run ingest.py`.

## Should be answered from the demo document
- What are the causes of secondary hypertension?
- What are the complications of hypertension?
- How is acute severe asthma treated?
- What are the causes of heart failure?
- ما هي أسباب الفشل القلبي؟  (Arabic question, English answer)
- ما هي مضاعفات ارتفاع ضغط الدم؟

## Should return "I don't know."
- What is the capital of France?          (not medical)
- What is the paracetamol dose for a 5-year-old child?   (medical, but not in the documents)

The second group matters most: the bot must never answer from the model's own memory.
