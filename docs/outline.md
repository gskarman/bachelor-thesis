# Thesis Outline

## 1. Introduction

- Background on AI-generated text and the growing need for detection
- Problem statement: opaque confidence scores lack interpretability
- Research questions
- Scope and limitations

## 2. Background

- Large Language Models and text generation
- Existing AI text detection methods (DetectGPT, watermarking, fine-tuned classifiers)
- Explainable AI (XAI) — definitions, taxonomies, and relevance to text classification
- Log-probability based classification

## 3. Method

- Single-token classification with log-probability extraction
- Calibration techniques
- Policy induction from labeled examples
- Explanation generation and faithfulness testing
- Models: gpt-oss 20B, Qwen2.5 32B
- Baselines: DetectGPT, fine-tuned RoBERTa
- Dataset: HC3

## 4. Results

- Detection performance (F1, AUROC, calibration error)
- Explanation faithfulness (feature-masking experiments)
- Comparison with baselines

## 5. Discussion

- Interpretation of results
- Limitations and threats to validity
- Ethical considerations

## 6. Conclusion

- Summary of findings
- Contributions
- Future work

## References
