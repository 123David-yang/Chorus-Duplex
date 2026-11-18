# Semantic review

Semantic review is optional. It uses the configured script LLM; it does not listen
to audio, retrieve authoritative sources, or establish that external claims are true.
An accepted script is not a guarantee of factual correctness or a perfect quality score.

`scripts.text.judge.review.review_dialogue` requires eight dimensions, including
factuality. Missing dimensions quarantine the report; factuality below 4 rejects it.
`review_scale_draft` requires five dimensions, including factuality, and rejects
factuality below 4 even when the other scores are high. Existing saved reviews are
not rewritten or assigned a default factuality score.

## Focused recall checks

For sampled drafts, `scripts.text.judge.attribution.review_attribution(draft)`
examines narrow Chinese and English second-person recall patterns such as
“你刚才说” and “you said”. Each task includes only the context preceding that
recall, with speaker and addressee metadata. The model must cite an earlier source
turn. Evidence that contradicts the actual source speaker or recipient is rejected
as invalid. A detected ownership mismatch rejects the sample; unresolved evidence
quarantines it. No candidates returns `not_applicable`, not `accept`.

```python
import json
from scripts.config import load_env
from scripts.text.judge.attribution import review_attribution

load_env()  # Use your private configuration; never commit credentials.
with open("sample.json", encoding="utf-8") as source:
    draft = json.load(source)
report = review_attribution(draft)  # Explicit, potentially billable LLM call.
print(report["decision"])
```

This is a sampled QA aid, not an exhaustive reference resolver. It may miss
indirect references or misread quoted speech. Ambiguous group recipients are
not accepted as proven ownership. Malformed or contradictory evidence raises
`ValueError`; callers should retain the sample for inspection rather than accept it.
There are no automatic repair, retry, production hooks, or extra per-item calls.

Small Chinese/English contrast tests distinguish a human's restriction mistakenly
addressed to the assistant from the same recall addressed to the correct human.
They do not establish broad accuracy: one Chinese negative case yielded invalid
evidence, and whole-dialogue aggregate scoring previously missed the same error.
Use independent inspection and representative samples alongside these checks.
