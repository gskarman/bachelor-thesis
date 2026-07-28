# Inlämning 6 — Rapport till examinator — hand-in checklist

**Deadline:** Thursday 7 May 2026, 19:00 CEST (today, ~11h away as of 08:23).
**Where:** Canvas → https://canvas.kth.se/courses/59476/assignments/352910
**File types accepted:** `doc` and `pdf` only (per the assignment page). **Recommend exporting from Google Docs as PDF** — avoids any layout/font ambiguity Canvas might silently re-render.
**Points:** 0 (this is the examiner-feedback round; Inl. 7 on 10 Jun is the final).
**Doc:** Kexjobbsuppsats — https://docs.google.com/document/d/1-0yFrBHGKaZgofwFds5WdUvnNcwN8nQF3hrO-CHesyw/edit

Canvas state verified in this session: assignment page reachable, no new standard-track announcements since 2026-03-06; the 17-Apr post is project-track-only and ignorable for our track. No new instructions added to the Inl. 6 page itself ("No additional details were added for this assignment.").

---

## 1. Must-fix in the doc before export

These are issues I caught comparing the current Google Doc against `logs/RUNS.md`, `docs/decisions.md`, and the Kexjobbsramar spec.

- [ ] **Citation for GLTR is wrong.** §2.2 paragraph 1 reads: *"GLTR was the visualisation-first version that showed token-rank histograms and let a human eyeball things (Guo et al. 2023)."* — GLTR is Gehrmann, Strobelt & Rush 2019, not Guo 2023 (Guo 2023 is the HC3 paper). The reference list already has Gehrmann et al. 2019 listed, so the in-text citation just needs to be swapped to `(Gehrmann et al. 2019)`.

- [ ] **Year mismatch on Zhu et al. (Hypotheses-to-Theories).** Body cites it as `(Zhu et al. 2024)` in §1 ("Hypotheses-to-Theories (Zhu et al. 2024)") and in §5.2 ("Hypotheses-to-Theories (Zhu et al. 2024)"). Reference list at the bottom lists it as `Zhu et al. 2023`. Pick one — ICLR 2024 / arXiv 2023 is the actual paper, so either reference works as long as the in-text year matches the entry. Easiest: change the reference-list year to 2024 (matches the venue and your in-text usage).

- [ ] **Faithfulness sample size — n=100 vs n=300.** §4.3 text and Table 5 both say `n = 100`. But `decisions.md` D12 says the scaled run used `sample_size=300`, and `RUNS.md` 2026-04-26 calls it "3 policies × 300 = 900 classifications". The actual `2026-04-26T20-23-43_2f80b2` row also shows `n=100 (test)` in the metrics column header. Verify which one was actually run by reading `logs/runs/2026-04-26T20-23-43_2f80b2/` and align the thesis text with reality. (If it was n=100, decisions.md D12 needs a tiny correction; if n=300, the §4.3 prose and the Table 5 `n logprob-valid` row need updating.)

- [ ] **Title-page placeholder text.** Doc currently opens with `# test 2` above the real title — that's leftover from a manual edit. Remove before exporting.

- [ ] **Stray `\` characters in body.** Several places have `\=` or `\>` or `0,5 \-` — these are escape artefacts from a markdown→Doc paste (e.g. *"F0.5 \= 0.934"*, *"+9.6 nat"* OK, *"\\~10,000 words"* in operating brief is fine because it's not in the report). Search the doc for `\=`, `\<`, `\>`, `\-`, `\~`, `\#` and clean up.

- [ ] **Image references that may not exist.** Body has three `![][image1]`, `![][image2]`, `![][image3]` markers (cover image and two figures referenced from §4.1 and §4.3). Confirm those images render properly in Google Docs view *and* in the exported PDF — if any are broken, either embed the actual figures from the run logs or remove the reference.

## 2. Spec compliance check

Hard rules from operating-brief §5 (Kexjobbsramar) — for an examiner round these are the bar:

- [ ] **Length: ~10,000 words / ~20 pages new ACM template** (per Henrik 20-Feb announcement; the older "10–12 pages" line refers to the old template). Run a word count on the Doc — Tools → Word count — and note the number. The current draft looks ~5–6k words by eye, so this is the most likely flag from an examiner.
- [x] **Abstract in BOTH SV and EN.** Sammanfattning (SV) and Abstract (EN) are both present. Each ~300 words target — eyeball check, both look in range.
- [x] **Section structure present:** Introduction · Background · Method · Results · Discussion · Conclusion · References. Order matches spec.
- [x] **References in ACM-ish format, ≥10 peer-reviewed.** 13 entries, mostly conference/journal — within the 10–15 target. (Two stylistic nits: ACM format expects "Authors. Year. Title. *Venue*..." — your entries use "Authors. Year. Title. In Proc. Venue." which is close. Don't refactor today; flag for Inl. 7.)
- [x] **Conclusion ≤ 300 words.** Section 6 looks within range.
- [ ] **Plagiarism.** Paste your final text into Canvas's Turnitin-equivalent (it runs automatically on submission) — but skim once for any direct quotes that aren't in quotation marks. The thesis doesn't seem to have any but worth a once-over.

## 3. Inl. 6 NOT required (deferred to Inl. 7 / 10 Jun)

Just so you don't waste time on these today:

- KTH cover page (intra.kth.se/kth-cover/) — Inl. 7 only.
- Publiceringsmedgivande (signed PDF) — Inl. 7 only.
- Email of title-in-other-language to supervisor + oviberg@kth.se — Inl. 7 only.
- DIVA registration — administrator does it after final approval.

## 4. Export & submit (the actual hand-in steps)

1. [ ] **Final read-through pass** in Google Docs. Watch especially for: (a) the citation/year fixes above, (b) any half-finished sentences from a recent edit session, (c) the §4.1 figure reference rendering correctly.
2. [ ] **Export as PDF.** Google Docs → File → Download → PDF Document (.pdf). Save with a sensible filename, e.g. `Kexjobbsuppsats-Skarman-Inl6-2026-05-07.pdf`.
3. [ ] **Open the exported PDF** and skim the first and last pages plus one figure page to confirm nothing was mangled by the export.
4. [ ] **Go to Canvas** → https://canvas.kth.se/courses/59476/assignments/352910 → click **Start Assignment** (top right).
5. [ ] **Upload the PDF** under "File upload". Click **Submit Assignment**.
6. [ ] **Confirm submission** — Canvas shows a green confirmation banner and the assignment moves to "Submitted". Take a screenshot for your records. (Canvas does sometimes silently reject if the file type isn't accepted — confirm visually.)
7. [ ] **Note the submission** in Notion: mark the "Submit report to examiner (Inlämning 6)" task (https://www.notion.so/346be2f61d9b81d38757cd7971684b69) as Done.

## 5. Nice-to-have if time permits (not blocking for Inl. 6)

Since Inl. 6 is the round before Inl. 7 (final on 10 Jun), each examiner comment you can pre-empt now is one less thing to fix in 5 weeks:

- [ ] **Hit the 10k-word target** if the current draft is short. The likeliest section to expand without padding: §2 Background (more on watermarking, more on RoBERTa-baseline numbers), §5.3 Method discussion (cross-comparison with ProTeGi/HtT on HC3).
- [ ] **DetectGPT / RoBERTa concrete baseline numbers in §4.1.** Currently you cite them but don't give a number-vs-number comparison; an examiner is likely to ask. Even one column "DetectGPT (cited from Mitchell 2023, HC3)" in Table 1 with a number would land.
- [ ] **Library lunch webinar at 12:15 today** — "Kommunicera dina resultat med Powerpoint och posters", Zoom, no signup. Useful for Tekniska prep on 23 May, not for today's submission.

---

**Bottom line:** the draft is in honest shape for an examiner round — the headline numbers (F0.5 0.934 / AUROC 0.982 / ECE 0.013 / faithfulness Δlabel 0.490 / Δlp +9.6 nats) all match `logs/RUNS.md` exactly. The two real risks are (1) the GLTR/Zhu citation slips above and (2) the n=100/n=300 faithfulness inconsistency. Fix those, export to PDF, upload, done.
