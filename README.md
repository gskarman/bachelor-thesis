# Explainable AI Text Detection

Bachelor's thesis at KTH Royal Institute of Technology (DM128X, Media Technology).

## About

This project explores whether Large Language Models can serve as both accurate and explainable detectors of AI-generated text. Current detection tools output opaque confidence scores without any rationale, creating a trust deficit for educators, editors, and other stakeholders who need to act on these results.

The approach uses single-token classification with log-probability extraction and calibration, combined with policy induction from labeled examples to generate auditable classification rules. Two local models (gpt-oss 20B and Qwen2.5 32B) are evaluated against established baselines like DetectGPT and fine-tuned RoBERTa on the HC3 dataset.

## Structure

```
bachelor-thesis/
├── code/       # Source code and experiments
├── docs/       # Thesis outline, notes, and documentation
├── logs/       # Experiment logs and results
└── README.md
```

## Resources

- [Linear Project Board](https://linear.app/namraks-consulting/project/bachelors-work-or-kth-82aefecb2647)
