# Production drafting modes

The production runner accepts `--draft-mode direct`, `interactive`, or `roleplay`.
All modes use the same planned event relationships and delivery schema.

| Mode | Writing strategy | LLM calls per successful draft attempt |
|---|---|---|
| `direct` (default) | Write the complete planned dialogue | 1 |
| `interactive` | Write two consecutive chunks, passing speaker/address metadata with the first chunk | 2 |
| `roleplay` (experimental) | Write one scheduled role turn at a time using preceding shared history | One per non-fixed turn |

Roleplay uses role-conditioned prompts to the configured LLM, not separate autonomous
services. Fixed lines consume no writing call. Private choice preferences are limited
to the current role. A backchannel role sees its host's audible anchor prefix, while
later ordinary turns can see the completed host. Invalid output shape stops subsequent
turn calls. The generator's ordinary draft retry can still repeat a failed attempt.

Roleplay is not the production default or an established quality improvement. One
22-turn English trial used 22 calls and about 61,000 tokens and still showed ownership
and continuity errors. Growing history increases input costs. Use it for comparison
and inspect results before synthesis. Cached drafts cannot be switched between modes;
use a separate output directory for a different mode.

## Optional experimental acknowledgement-anchor review

Set `CHORUS_ALIGN_BACKCHANNELS=1` in the task process to enable one focused review of
new drafts' acknowledgement groups. It chooses an original host-text prefix sufficient
to support all acknowledgements in that group. Code checks exact prefix membership,
keeps at least eight characters after it and applies changes only if every group has
valid output. An impossible or malformed result rejects the candidate.

```powershell
$env:CHORUS_ALIGN_BACKCHANNELS = '1'
python -m scripts.production.scale_moss --root D:\data\chorus-new --hours 1 --draft-mode direct
```

The check adds at most one model-call attempt to a candidate with backchannels.
After an anchor-review attempt, a failure is retained rather than triggering another
paid full rewrite. Existing structural retries before that step remain. Drafts without
backchannels make no alignment call; already cached drafts and old deliveries are not
rewritten. This review does not fix ownership, factual claims or incomplete phrasing.

The review selects text boundaries, not word timestamps. The renderer measures the
selected prefix's synthesized duration and inserts the acknowledgement afterward.
Eight remaining characters do not guarantee enough audio for every listener; rendering
still validates placement. Small positive/negative Chinese/English checks support this
experiment, not broad semantic accuracy or listening-quality claims. It is disabled
unless explicitly enabled.
