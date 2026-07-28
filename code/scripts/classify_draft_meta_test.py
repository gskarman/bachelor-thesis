"""One-off: run a chunk of inl5-draft.md through the frozen-policy classifier
to see whether the classifier flags Gustav's bachelor-thesis prose (drafted by
Claude Opus 4.7) as AI-generated. Not a real evaluation — a meta-test.
"""

from __future__ import annotations

import math
from pathlib import Path

from aitd.classifier import classify, yes_no_prob_ai
from aitd.ollama_client import OllamaClient

REPO = Path(__file__).resolve().parents[2]
POLICY_PATH = REPO / "logs" / "policies" / "2026-04-23T23-06-25_b15d8f.md"

CONCLUSION = """\
Can an LLM detect AI-generated text both reliably and explainably? On the evidence assembled here, the answer is a qualified yes. With a small open-weights model (Gemma 4 E4B) and a ~150-word natural-language policy induced by a temperature-zero proposer/scorer loop on the HC3 train split, a single-token classifier reaches F0.5 = 0.942 / AUROC = 0.993 / ECE = 0.030 on the HC3 all test split — competitive with a four-times-larger default-prompt baseline. The qualifier matters: the comparison to published baselines is sensitive to HC3-subset choice and AI-source-LLM choice, and the per-domain breakdown shows precision deficits on open_qa and wiki_csai that are not eliminated by scale alone. On the explanation side, replacing the induced policy with an adversarially-inverted policy flips 46% of test labels and shifts the log-probability margin by +8.9 nats — direct behavioural evidence that the policy is causally load-bearing and not a post-hoc rationalisation. Taken together: the rule the model uses to flag a text is the rule a human auditor reads. That equivalence is what an educator confronting a flagged essay, an editor handling a flagged submission, or a reviewer adjudicating an authorship dispute actually needs — not "82% likely AI-generated", but "flagged because of these specific properties, and removing them would change the verdict". This thesis demonstrates that the equivalence is achievable on commodity hardware and measurable in practice; whether the resulting explanation is useful to a human reader, and whether the policy is portable across model families, are the open questions that frame the work that comes next.
"""


def load_policy() -> str:
    text = POLICY_PATH.read_text(encoding="utf-8")
    marker = "## Policy text"
    idx = text.index(marker) + len(marker)
    return text[idx:].strip()


def main() -> None:
    policy = load_policy()
    print(f"[meta-test] policy: {len(policy)} chars from {POLICY_PATH.name}")
    print(f"[meta-test] passage: {len(CONCLUSION)} chars (§6 Conclusion)\n")

    client = OllamaClient(model="gemma4:e4b")
    pred = classify(
        client,
        CONCLUSION,
        system_prompt=policy,
        return_logprobs=True,
        top_logprobs_k=20,
        temperature=0.0,
    )

    label_str = {1: "AI", 0: "human", -1: "other"}.get(pred.label, str(pred.label))
    print(f"raw_response: {pred.raw_response!r}")
    print(f"hard label  : {label_str}")
    if pred.logprobs:
        prob_ai = yes_no_prob_ai(pred.logprobs)
        top = sorted(pred.logprobs.items(), key=lambda kv: -kv[1])[:8]
        print(f"P(AI)       : {prob_ai:.4f}" if prob_ai is not None else "P(AI)       : n/a")
        print("top logprobs:")
        for tok, lp in top:
            print(f"  {tok!r:>12}  lp={lp:+.3f}  p={math.exp(lp):.4f}")
    else:
        print("(no logprobs returned)")


if __name__ == "__main__":
    main()
