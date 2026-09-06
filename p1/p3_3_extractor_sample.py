"""
P3.3a - Draw a stratified sample for validating the claim extractor.

The lexical extractor decides, for each (response, sensor) pair, whether the text
attributes an increasing, a decreasing or a stable trend to that sensor, or says
nothing directional about it. That decision carries the whole faithfulness
analysis and has never been checked against a human reading of the text.

This writes a sample to be adjudicated by hand. It is stratified over the
extractor's own verdict so that both directions of error can be measured: cases
it called directional, where a wrong call would inflate precision, and cases it
called unspecified or absent, where a missed claim would depress recall. Sampling
"absent" is deliberately light, since the extractor is right about most of them by
construction, but it is not zero.

    python p1/p3_3_extractor_sample.py            # writes the sample
    python p1/p3_3_extractor_validate.py          # scores it once annotated

Output: results_p1/p3_3_extractor_sample.csv, with an empty `human` column.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
P1 = EXP / "results_p1"
from analyze_explainability import SENSOR_PHYSICS, classify_mention  # noqa: E402

SEED = 20260906
QUOTA = {"increase": 40, "decrease": 40, "stable": 9, "unspecified": 30, "absent": 20}
CTX_BEFORE, CTX_AFTER = 120, 160


def contexts(text, sensor):
    """Every local window around a mention of the sensor, as the reader sees it."""
    out = []
    for m in re.finditer(rf"\b{sensor}\b", text, re.IGNORECASE):
        a = max(0, m.start() - CTX_BEFORE)
        b = min(len(text), m.end() + CTX_AFTER)
        out.append(("..." if a else "") + text[a:b].replace("\n", " ")
                   + ("..." if b < len(text) else ""))
    return out


def main():
    rows = []
    for path in sorted((EXP / "results").glob("traces_*.json")):
        traces = json.loads(path.read_text(encoding="utf-8"))
        for t in traces:
            txt = t.get("reasoning", "") or ""
            for sensor in SENSOR_PHYSICS:
                v = classify_mention(txt, sensor)
                rows.append({"file": path.name, "engine_id": t["engine_id"],
                             "sensor": sensor, "extractor": v,
                             "n_mentions": len(re.findall(rf"\b{sensor}\b", txt, re.I)),
                             "context": " || ".join(contexts(txt, sensor))[:1200]})
    df = pd.DataFrame(rows)
    print("extractor verdicts over the whole corpus:")
    print(df.extractor.value_counts().to_string())

    rng = np.random.default_rng(SEED)
    picks = []
    for verdict, q in QUOTA.items():
        sub = df[df.extractor == verdict]
        take = min(q, len(sub))
        picks.append(sub.iloc[rng.choice(len(sub), size=take, replace=False)])
    sample = pd.concat(picks).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "id", range(1, len(sample) + 1))
    sample["human"] = ""
    sample.to_csv(P1 / "p3_3_extractor_sample.csv", index=False, encoding="utf-8")
    print(f"\n{len(sample)} cases written to p3_3_extractor_sample.csv")
    print("Fill the `human` column with increase / decrease / stable / "
          "unspecified / absent, then run p3_3_extractor_validate.py")


if __name__ == "__main__":
    main()
