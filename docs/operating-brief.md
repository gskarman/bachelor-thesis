# Bachelor Thesis — Operating Brief

Reference document for Gustav Skarman's bachelor thesis at KTH. Intended to be loaded as context for agents working on thesis tasks. All canonical URLs, IDs, and rules are kept here — prefer this file over cached memory when agents run.

---

## 1. Quick summary

- **Course:** DM128X VT26 Examensarbete inom medieteknik, grundnivå (60801), VT 2026
- **Track:** Standard track (NOT the project/"KEX-projekt" track — ignore all "bara för projektgrupper" announcements)
- **Language:** TBD (report can be SV or EN; abstracts required in BOTH)
- **Next deadline:** Thursday 30 April 2026, 19:00 — Inlämning 5 (first draft to supervisor)
- **Final deadline:** Wednesday 17 June 2026, 23:59 — Godkända rapporter
- **Credits:** 15 hp, Grundnivå (kandidatexamen), Program Teknik, Ämne Medieteknik

## 2. Thesis context

**Topic.** Whether LLMs can reliably classify AI-generated text *and* produce human-interpretable explanations.

**Research questions.**
- **RQ1:** How does an LLM-based single-token classifier compare to established baselines (DetectGPT, fine-tuned RoBERTa) on standard AI-text detection benchmarks?
- **RQ2:** Can the classification decision be decomposed into human-interpretable explanations, and how faithful are those explanations to the model's actual decision mechanism?

**Method stack.**
- Classifier: LLM single-token classification with log-probability extraction and policy induction
- Models: gpt-oss 20B + Qwen2.5 32B
- Baselines: DetectGPT, fine-tuned RoBERTa
- Dataset: HC3
- Metrics: F1, AUROC, ECE + explanation faithfulness via feature masking

**Code:** https://github.com/gskarman/bachelor-thesis

## 3. People and contacts

- **Supervisor:** Jarmo Laaksolahti. Handledning room: 4618 (Flexistudio), D-huset plan 6, KTH.
- **Course responsible (announcements):** Roberto Bresin.
- **Course admin email (for titles / grade reporting):** oviberg@kth.se
- **Examiner:** See Canvas module "Handledare och examinatorer".

## 4. Full deadline schedule (standard track)

All times Europe/Stockholm. Submit on Canvas unless noted otherwise.

| # | Date | Time | Item | Type | Points | Notes |
|---|------|------|------|------|--------|-------|
| 1 | **Thu 30 Apr 2026** | 19:00 | Inlämning 5: Slutrapport till handledaren | File upload | 50 | First-draft milestone. Supervisor gives feedback before Inl. 6. |
| 2 | **Thu 7 May 2026** | 19:00 | Inlämning 6: Rapport till examinator | File upload | 0 | Revised version for examiner. |
| 3 | **Sat 23 May 2026** | 11:00–16:00 | Presentation / Demo / Examination at Tekniska museet | In-person event | — | First of two presentation occasions. |
| 4 | **Wed 3 Jun 2026** | 12:00–17:00 | Presentation / Examination / Demo at KTH Innovation | In-person event | — | Second presentation occasion. |
| 5 | **Wed 10 Jun 2026** | 19:00 | Inlämning 7: Slutlig rapport | File upload | 0 | Final report after presentation feedback. Plagiarism check runs. |
| 6 | **Fri 12 Jun 2026** | 19:00 | Inlämning 8: Reflektioner kring presentationerna | File upload | 0 | Written reflection on the two presentation events. |
| 7 | **Wed 17 Jun 2026** | 23:59 | Godkända rapporter! | Gate | 10 | Opens 18 May 17:00. Final approved-report checkpoint. |
| 8 | Undated | — | Medgivande av publicering i Diva | Admin | 0 | Individual (not per pair). See §9. |
| 9 | Undated | — | Godkänd slutgiltig rapport | Admin | 0 | Deposited in DIVA by administrator after final approval. |

## 5. Report structure (Kexjobbsramar)

Hard spec from the course. Every submission (Inl. 5, 6, 7) should follow this structure. Length targets assume 10–12 total pages.

| Section | Target length | What it must do |
|---------|---------------|-----------------|
| Abstract (SV + EN) | ~300 words each | Problem relevance, theory/related work, research question, method, result, discussion link-back. Both language versions are required. |
| Introduction | 1 page (incl. title, names, abstract) | Broad, accessible framing that narrows to the problem. Position the work. State the problem early. |
| Background (literature review) | 1–1.5 pages | Define key concepts. Present theory and related research systematically. End with purpose statement + research question. |
| Method & data collection | 0.5 page | Concise. How the question was investigated. Define variables/concepts. Justify choice vs. research question. |
| Results | 4–5 pages | Factual, no interpretation. Subheadings + visuals (tables, figures). Raw material can move to appendix. |
| Discussion | 2 pages | Summary → relate to theory/related work → method discussion → future research → explicit contribution. |
| Conclusion | ≤300 words | Short, practical/theoretical takeaway. |
| References | ~1 page | ACM format. ~15 peer-reviewed articles as the base. |

### Formalia (must always apply)

- **Length:** Kexjobbsramar states 10–12 pages in the ACM template. Course-responsible Henrik clarified in a Canvas announcement (20 Feb 2026, "Längd på uppsatsen/rapporten") that the practical target is **~10,000 words ≈ ~20 pages in the new ACM template** (KEX-mall, referenssystem & annat bra material). Treat ~10k words as the working target; the 10–12-page line refers to an older template. When in doubt, ask supervisor.
- **References:** ~10–15 peer-reviewed research articles (journals/conferences). Web pages and textbooks only as exceptions.
- **Language:** Swedish or English. Abstracts in BOTH.
- **Plagiarism:** Plagiarism check runs on every submission. See Plagiathandboken. Violations go to disciplinnämnden.
- **Pairs:** Report written with one course mate. Solo requires dispensation and same total scope.

### Section-specific rules

- **Abstract:** Don't paste from PDF (risk of broken line breaks). Copy from original doc.
- **Introduction:** Opening hook ("anslaget") matters. Write so a general reader is drawn in. State the problem early.
- **Background:** Systematic — theory sub-section + related-work sub-section. Ends with purpose + RQ.
- **Method:** Literature reading is NOT a method (goes in theory). Define variables clearly. For experimental studies: define independent and dependent variables.
- **Results:** Purely factual. Interpretation goes in Discussion.
- **Discussion:** Must include a method discussion ("was this the right method?") and contribution (to field + practice). Briefly address future work.
- **Conclusion:** Max 300 words.

## 6. Final-submission-only requirements (Inl. 7 and later)

The following apply at the final-report stage, not the first draft:

- **KTH cover page** generated via http://intra.kth.se/kth-cover/ — do NOT fill in ISSN, TRITA, or ISRN.
  - Uppsatsnivå: Självständigt arbete på grundnivå (kandidatexamen)
  - Högskolepoäng: 15
  - Utbildningsprogram: Teknik
  - Ämne/kurs: Medieteknik
  - Title in the same language as the report. Use `<br/>` if long.
  - Names in mixed case ("Gustav Skarman", not "GUSTAV SKARMAN").
- **Publiceringsmedgivande:** individually signed (physical signature, not digital), scanned (PDF or image), filename = firstname-lastname.
- **Title in other language:** If report is SV, email EN title to supervisor + oviberg@kth.se. If report is EN, send SV title.
- **DIVA:** A study administrator registers the approved report in DIVA. The student does NOT submit to DIVA directly.

## 7. Canvas resources (canonical URLs)

- **Course home:** https://canvas.kth.se/courses/59476
- **Assignments list:** https://canvas.kth.se/courses/59476/assignments
- **Announcements:** https://canvas.kth.se/courses/59476/announcements
- **Calendar:** https://canvas.kth.se/courses/59476/calendar
- **Modules:** https://canvas.kth.se/courses/59476/modules

### Assignment pages (Inlämning N)

- Inlämning 5 (30 Apr): https://canvas.kth.se/courses/59476/assignments/352909
- Other Inlämningar: navigate from Assignments list above (IDs not yet captured).

### Key module pages

- **Kexjobbsramar** (report spec): https://canvas.kth.se/courses/59476/pages/kexjobbsramar
- **Instruktioner för slutlig rapport** (final-report rules): https://canvas.kth.se/courses/59476/pages/instruktioner-for-slutlig-rapport
- **KEX-mall, referenssystem & annat bra material** (ACM template etc.): https://canvas.kth.se/courses/59476/pages/kex-mall-referenssystem-and-annat-bra-material
- **FORMALIA_Kexjobb på Borggården VT24 student.pdf** (example): https://canvas.kth.se/courses/59476/modules/items/1292906
- **Publiceringsmedgivande.pdf:** https://canvas.kth.se/courses/59476/modules/items/1437205
- **Intro_DM128X_2026_01_13.pdf** (intro lecture): https://canvas.kth.se/courses/59476/modules/items/1393612
- **Hitta källor VT-26 DM128x.pdf** (source-finding guide): https://canvas.kth.se/courses/59476/modules/items/1398740

### Announcements to re-check

- **VIKTIGA DATUM:** https://canvas.kth.se/courses/59476/discussion_topics/543633
- **Uppdaterad titel/beskrivning samt plats:** https://canvas.kth.se/courses/59476/discussion_topics/545280 (unread — project-track, may be ignorable)

## 8. Google Drive — KEXET folder

Shared from `gustav.skarman@gmail.com` (not `gustav@namraks.com`). All thesis writing lives here.

- **Folder:** [KEXET](https://drive.google.com/drive/folders/1cXsw8n-ccHSZvltzFVC_OZfXK2tRc4nx) (ID: `1cXsw8n-ccHSZvltzFVC_OZfXK2tRc4nx`)

| File | Role | ID |
|------|------|----|
| [Kexjobbsuppsats](https://docs.google.com/document/d/1-0yFrBHGKaZgofwFds5WdUvnNcwN8nQF3hrO-CHesyw/edit) | **Primary thesis report document** — this is what gets submitted. Currently near-empty (just "Utkast") as of 14 Apr 2026. | `1-0yFrBHGKaZgofwFds5WdUvnNcwN8nQF3hrO-CHesyw` |
| [Kexjobbsspecifikation](https://docs.google.com/document/d/1vgsi4AlcWnbH0WxjGzK6R-DepT8yPNjx6-centHyMm0/edit) | Inlämning 4 specification. Source material for Introduction, Background, Method. | `1vgsi4AlcWnbH0WxjGzK6R-DepT8yPNjx6-centHyMm0` |
| [Kexjobsramar](https://docs.google.com/document/d/123uCwRTB5xLfykX06t6-VwBveaPcveqfK0IuGLQ716M/edit) | Local copy of the report spec. | `123uCwRTB5xLfykX06t6-VwBveaPcveqfK0IuGLQ716M` |
| [Utkast ide och metod dokument](https://docs.google.com/document/d/1jy3E-lrKzG0MuhHbJba6rqcPr3HM79o9dQE2E_3cV7I/edit) | Early idea/method draft. | `1jy3E-lrKzG0MuhHbJba6rqcPr3HM79o9dQE2E_3cV7I` |
| [Kandidatexamensarbete föreläsning referenser](https://docs.google.com/document/d/1Yotf66cJZ-1-gHbNCIdtAxvoVFNCNJqXULO2lmezpAA/edit) | Seed reference list from the intro lecture. | `1Yotf66cJZ-1-gHbNCIdtAxvoVFNCNJqXULO2lmezpAA` |

## 9. Notion — project and tasks

### Project page

- **Bachelors Work | KTH** — https://www.notion.so/33ebe2f61d9b8161b224f1995ef0dee1
  - Area: Learning
  - Status: Active
  - Start: 2026-04-01 · End: 2026-06-10
  - Contains the milestone table and RQ summary.

### Tasks database (data source ID: `56dbc92c-c322-470d-89a1-5d200bec440e`)

Task schema (for agents adding or updating tasks):

- `Task` (title, required)
- `Status` ∈ {Inbox, Next, In Progress, Waiting, Done}
- `Priority` ∈ {Urgent, High, Medium, Low}
- `Effort` ∈ {15 min, 30 min, 1 hour, 2+ hours}
- `Energy` ∈ {High, Medium, Low}
- `Type` ∈ {Build, Research, Write, Admin, Review, Learn, Fix, Design}
- `Context` multi-select ⊂ {deep-work, quick-win, errand, meeting, review}
- `Project` — relation to Projects data source (use the project page URL)
- `Notes` — short text
- `Due Date`, `Start Date`, `Completed` — expanded as `date:<name>:start`, `date:<name>:end`, `date:<name>:is_datetime`
- `Blocked By` — relation to other tasks
- `Related Notes` — relation to Notes database (`collection://5c2152c1-a2b2-4086-b446-64e9b3d872e4`)

### Thesis tasks (all linked to the project above)

| Task | Due | URL |
|------|-----|-----|
| Submit draft report to supervisor (Inlämning 5) | 2026-04-30 | https://www.notion.so/33ebe2f61d9b813389aed30d80cbf4b8 |
| Submit report to examiner (Inlämning 6) | 2026-05-07 | https://www.notion.so/346be2f61d9b81d38757cd7971684b69 |
| Presentation/Demo at Tekniska (examination) | 2026-05-23 11:00 | https://www.notion.so/346be2f61d9b81f8b2fae892cbebfced |
| Presentation at KTH Innovation | 2026-06-03 12:00 | https://www.notion.so/346be2f61d9b8174aed2f34c19b212fb |
| Submit final report (Inlämning 7: Slutlig rapport) | 2026-06-10 | https://www.notion.so/346be2f61d9b81a5b8eee07af705fba8 |
| Submit reflections on presentations (Inlämning 8) | 2026-06-12 | https://www.notion.so/346be2f61d9b8192ab4de8f1133a322b |
| Godkända rapporter (final checkpoint) | 2026-06-17 | https://www.notion.so/346be2f61d9b81a48cacf40e9ad71ce8 |
| Publiceringsmedgivande till DIVA (sign & upload) | undated (~10 Jun) | https://www.notion.so/346be2f61d9b81529ee2d438443a162b |
| Create KTH cover page for final report | undated (~10 Jun) | https://www.notion.so/346be2f61d9b812298f1dec47b46fc88 |

## 10. Upcoming deadline — Inlämning 5 (first draft)

**Hard facts.**
- Due: Thursday 30 April 2026, 19:00 CEST
- Submit via: Canvas file upload at https://canvas.kth.se/courses/59476/assignments/352909
- Points: 50
- Labeled "Slutrapport till handledaren" but in practice the supervisor treats this as a first-draft feedback round before Inlämning 6 on 7 May.

**State as of 18 April 2026.**
- The thesis document (`Kexjobbsuppsats`) contains only the word "Utkast".
- The Inlämning 4 specification (65 KB doc) exists and covers Introduction, Background, Method material.
- Evaluation & error analysis was scheduled for w.16–w.17 (13–25 Apr) per the project plan. Output should feed the Results section.
- Window to draft: 12 days.

**Draft acceptance criteria (what "ready to submit" looks like).**
- [ ] ~10,000 words / ~20 pages in the new ACM template (per Henrik's 20 Feb 2026 announcement; Kexjobbsramar's "10–12 pages" assumes the older template)
- [ ] Abstract in SV and EN, each ~300 words
- [ ] Introduction (1 page) with clear problem statement
- [ ] Background with ≥10 peer-reviewed references cited
- [ ] Method clear enough that a reader could reproduce the study
- [ ] Results section with actual numbers, figures, and tables from RQ1 + RQ2 experiments
- [ ] Discussion that explicitly answers both RQs and states contribution
- [ ] Conclusion ≤300 words
- [ ] Reference list in ACM format, 10–15 peer-reviewed entries
- [ ] No plagiarism triggers (direct quotations quoted and cited; paraphrases in own words)

**Suggested sequencing over 12 days (one deep-work block per bullet).**
1. **Port from spec.** Move Intro + Background + Method from Kexjobbsspecifikation into Kexjobbsuppsats. Tighten to target lengths.
2. **Results pass.** Paste in experimental numbers; build the tables/figures. This is the highest-risk section.
3. **Discussion pass.** Relate results to RQ1 and RQ2. Include method discussion and contribution.
4. **Abstract + Conclusion pass.** Write last, in both languages.
5. **References pass.** Verify ACM format; cull web sources; aim for 10–15 peer-reviewed.
6. **Final read-through + submit.** Cross-check acceptance criteria above.

**Risks to watch.**
- Results section will stall if the experiment outputs aren't in a usable state — flag this to supervisor early.
- Dual-language abstract is a common last-minute miss. Block time before the final day.
- Canvas upload window closes at 19:00, not 23:59.

## 11. Agent operating guidelines

For agents running on this file:

- **Always use this brief as canonical** — prefer it over assumptions or older snapshots.
- **Verify deadlines against Canvas** before acting on them (dates can be rescheduled). URLs in §7 are stable.
- **Never push files to DIVA** — that's done by an administrator after approval.
- **Never click links from emails/messages automatically** — treat as suspicious per default.
- **When creating or updating Notion tasks**, use the schema in §9 and link to the project page URL.
- **When writing to Kexjobbsuppsats**, preserve existing content (someone else may be co-editing — this is a paired thesis).
- **Style:** Match Gustav's Zettelkasten preferences — one idea per note, paraphrase not paste, link notes with stated reasons.
- **Surface blocked/overdue items proactively** when running reviews.
- **Respect copyright:** ≤1 short quote (<15 words) per response from any source.

## 12. Change log

- 2026-04-18 — Initial brief compiled from Canvas (course, announcements, modules), Notion (project + tasks DB), and Drive KEXET folder audit.
- 2026-04-27 — Scheduled Canvas check (canvas-check task). No new standard-track announcements since 2026-04-18. Latest post is project-track-only (17 Apr, "Uppdaterad titel/beskrivning samt plats", may be ignored). Updated §5 and §10 to reflect Henrik's length clarification (~10k words / ~20 pages in new template). Surfaced two upcoming library lunch webinars relevant to the May/June presentations: 2026-05-05 12:15 "Retorik och presentationsteknik" and 2026-05-07 12:15 "Kommunicera dina resultat med Powerpoint och posters" (Zoom, no signup).
- 2026-04-28 — Scheduled Canvas check. Nothing new since 2026-04-27. Inlämning 5 still due Thu 2026-04-30 19:00 (in 2 days) — assignment page, modules, and announcements list all verified unchanged. Latest standard-track announcement remains the 2026-03-06 library workshops post; only post since is project-track-only (17 Apr, ignorable).
- 2026-04-29 — Scheduled Canvas check FAILED to read content: canvas.kth.se redirected to KTH ADFS sign-in (browser session expired). No new info captured. Notion not updated. **Action for Gustav:** sign into Canvas in the controlled Chrome instance so tomorrow's check (and the next manual one before tomorrow's 19:00 Inl. 5 deadline) can proceed. Inlämning 5 deadline unchanged per local context: Thu 2026-04-30 19:00 (~25h away).
- 2026-04-30 — Scheduled Canvas check (canvas-check task). **Sign-in restored** — all course surfaces reachable. Nothing new since 2026-04-28: latest standard-track announcement is still 2026-03-06 library workshops; only post since is the 17 Apr project-track-only "Uppdaterad titel/beskrivning samt plats" (ignorable). Inlämning 5 page verified at https://canvas.kth.se/courses/59476/assignments/352909 — **due TODAY Thu 2026-04-30 by 19:00**, 50 pts, file upload, no schedule changes. Assignments list and Modules tree both unchanged from prior snapshots. Notion project page updated with status comment.
- 2026-05-04 — Scheduled Canvas check FAILED to read content: canvas.kth.se redirected to KTH ADFS sign-in (browser session expired, same failure mode as 2026-04-29). No new info captured. **Action for Gustav:** sign into Canvas in the controlled Chrome instance so the next scheduled check can proceed — and one manual check before Thu 2026-05-07 19:00 (Inlämning 6 deadline, ~3 days away) is recommended. Notion project page updated with status comment. Inlämning 6 deadline unchanged per local context: Thu 2026-05-07 19:00.
- 2026-05-05 — Scheduled Canvas check FAILED to read content: canvas.kth.se still redirects to KTH ADFS sign-in (third consecutive failure since 2026-04-30). No new info captured. **Action for Gustav (now urgent):** Inlämning 6 is due Thu 2026-05-07 19:00 (~2 days away). Sign into Canvas in the controlled Chrome instance and do one manual check of https://canvas.kth.se/courses/59476/announcements before the deadline — there have been zero successful auto-checks in the run-up to Inl. 6. Notion project page updated with status comment.
- 2026-05-06 — Scheduled Canvas check FAILED to read content: canvas.kth.se still redirects to KTH ADFS sign-in (fourth consecutive failure since 2026-04-30). No new info captured. **Action for Gustav (URGENT — last day before deadline):** Inlämning 6 due tomorrow Thu 2026-05-07 19:00 (~30h away). Sign into Canvas in the controlled Chrome instance and do a manual check of https://canvas.kth.se/courses/59476/announcements before submitting — every auto-check in the Inl. 6 run-up has failed. Notion project page updated with status comment.
- 2026-05-07 — Scheduled Canvas check FAILED to read content: canvas.kth.se still redirects to KTH ADFS sign-in (fifth consecutive failure since 2026-04-30). No new info captured. **Action for Gustav (URGENT — DEADLINE TODAY):** Inlämning 6 due today Thu 2026-05-07 19:00 (~11h away). Sign into Canvas in the controlled Chrome instance and do a manual check of https://canvas.kth.se/courses/59476/announcements before submitting — every auto-check in the Inl. 6 run-up has failed. Notion project page updated with status comment.
- 2026-05-08 — Scheduled Canvas check (canvas-check task). **Sign-in restored** — all course surfaces reachable after 5 consecutive ADFS failures. Nothing new on the standard track since 2026-04-28: latest standard-track announcement is still the 2026-03-06 library workshops post; only post since is the 17 Apr project-track-only "Uppdaterad titel/beskrivning samt plats" (ignorable). Inlämning 6 (due yesterday Thu 2026-05-07 19:00) now in "Past assignments" with status "Not Yet Graded" — Canvas indicates a submission was received and is awaiting examiner grading. Inlämning 5 also "Not Yet Graded". Assignments list and Modules tree unchanged from prior snapshots. Notion project page updated with status comment. Next milestone: Sat 2026-05-23 11:00 — Presentation/Demo at Tekniska museet (15 days away).
- 2026-05-11 — Scheduled Canvas check FAILED to read content: canvas.kth.se redirected to KTH ADFS sign-in (browser session expired again, first failure since 2026-05-08). No new info captured. **Action for Gustav:** sign into Canvas in the controlled Chrome instance so the next scheduled check can proceed. Not urgent — next milestone (Presentation/Demo at Tekniska museet) is Sat 2026-05-23 11:00, 12 days away — but one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation week is recommended to catch any room/time/format updates. Notion project page updated with status comment.
- 2026-05-12 — Scheduled Canvas check FAILED to read content: canvas.kth.se still redirects to KTH ADFS sign-in (second consecutive failure since 2026-05-08; both `/announcements` and `/modules` confirmed redirecting). No new info captured. Notion update skipped — the Notion MCP connector is unauthenticated in this scheduled run, so no project-page status comment was posted. **Action for Gustav:** (1) sign into Canvas in the controlled Chrome instance so the next scheduled check can proceed; (2) one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation week is recommended to catch any room/time/format updates for Tekniska museet. Not urgent — next milestone (Presentation/Demo at Tekniska museet) is Sat 2026-05-23 11:00, 11 days away.
- 2026-05-14 — Scheduled Canvas check (canvas-check task). Canvas reachable today via the **public course preview** (no ADFS redirect). The logged-in `/announcements` list page renders empty for anonymous viewers, but individual discussion topics and the **Canvas REST API** (`/api/v1/announcements?context_codes[]=course_59476`) return the full announcement feed without auth. Pulled the full 2026-01-01 → 2026-05-15 window. Nothing new on the standard track since 2026-03-06: latest standard-track post is still "Biblioteket erbjuder Workshops"; the 3 April announcements (id 545280, 543633, 543632) and 25 March (id 542594) are all PROJECT-track only ("LÄS DETTA OM DU GÖR PROJEKT: …"), ignorable. Modules tree and Assignments list unchanged from prior snapshots. Note: assignment grading state is not visible in the public preview, so Inl. 6 "Not Yet Graded" status cannot be re-confirmed today (was confirmed 2026-05-08). Next milestone: Sat 2026-05-23 11:00 — Presentation/Demo at Tekniska museet, 9 days away. Notion project page updated with status comment.
- 2026-05-15 — Scheduled Canvas check (canvas-check task). Canvas REST API reachable without auth (same path as 2026-05-14). Announcement feed unchanged: 11 announcements total, latest standard-track post still 2026-03-06 "Biblioteket erbjuder Workshops"; the four posts since (25 Mar / 1 Apr × 2 / 16 Apr) are all PROJECT-track only and ignorable. Assignments list verified unchanged — all 12 entries match prior snapshot (Inl. 1–8 + Godkända rapporter + Diva + Slutgiltig rapport); due dates intact. Modules tree verified unchanged — 8 modules, 23 items total, all titles match prior snapshot. Calendar events (course-scoped, 2026-05-15 → 2026-06-30) empty. Note: grading state still not visible in public preview, so Inl. 5/6 status cannot be re-confirmed (last confirmed 2026-05-08: "Not Yet Graded"). Next milestone: Sat 2026-05-23 11:00 — Presentation/Demo at Tekniska museet, **8 days away**. Notion project page updated with status comment.
- 2026-05-18 — Scheduled Canvas check FAILED. All Canvas surfaces tested redirect to ADFS sign-in / require auth: `/courses/59476` and `/announcements` redirect to login.ug.kth.se; the REST API (`/api/v1/announcements?context_codes[]=course_59476`) now returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`. The anonymous-API path that worked on 2026-05-14 and 2026-05-15 is **closed** today — either browser session expired or Canvas tightened API auth. No new info captured. **Action for Gustav:** sign into Canvas in the controlled Chrome instance so the next scheduled check can proceed, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation week to catch any room/time/format updates for Tekniska museet. Notion project page updated with status comment. Next milestone: Sat 2026-05-23 11:00 — Presentation/Demo at Tekniska museet, **5 days away**.
- 2026-05-19 — Scheduled Canvas check FAILED (second consecutive failure since 2026-05-15). Same pattern as 2026-05-18: `/courses/59476` and `/announcements` redirect to ADFS (login.ug.kth.se); REST API (`/api/v1/announcements?context_codes[]=course_59476`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`. The anonymous-API path that worked on 2026-05-14/15 remains closed. No new info captured. **Action for Gustav (getting time-sensitive):** Presentation/Demo at Tekniska museet is Sat 2026-05-23 11:00, **4 days away**. Sign into Canvas in the controlled Chrome instance so the next scheduled check can proceed, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation week to catch any room/time/format updates. Notion project page updated with status comment.
- 2026-05-20 — Scheduled Canvas check FAILED (third consecutive failure since 2026-05-15). Same pattern as 2026-05-18 / 2026-05-19: `/courses/59476` and `/announcements` redirect to ADFS (login.ug.kth.se); REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-05-25`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`. The anonymous-API path that worked on 2026-05-14/15 remains closed. No new info captured. **Action for Gustav (now time-sensitive):** Presentation/Demo at Tekniska museet is Sat 2026-05-23 11:00, **3 days away**. Sign into Canvas in the controlled Chrome instance so the next scheduled check can proceed, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation to catch any room/time/format updates for Tekniska museet. Notion project page updated with status comment.
- 2026-05-21 — Scheduled Canvas check FAILED (fourth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-05-20: the logged-in Chrome session at `/courses/59476/announcements` redirects to ADFS (login.ug.kth.se "Sign In" page, confirmed via page text), and the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-05-25`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}` when fetched directly in-browser. The anonymous-API path that worked on 2026-05-14/15 remains closed. No new info captured. **Action for Gustav (URGENT — presentation in 2 days):** Presentation/Demo at Tekniska museet is Sat 2026-05-23 11:00, **2 days away**. Sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation to catch any room/time/format updates for Tekniska museet — there have been zero successful auto-checks in the run-up to this presentation. Notion project page updated with status comment. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-05-22 — Scheduled Canvas check FAILED (fifth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-05-21: logged-in Chrome at `/courses/59476/announcements` and `/courses/59476` both redirect to ADFS (login.ug.kth.se "Sign In" page, confirmed via page text); the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-05-25`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`; the public course-preview path that worked on 2026-05-14/15 is also closed today. No new info captured. Notion **was** reachable this run — project page updated with a status comment. **Action for Gustav (URGENT — presentation TOMORROW):** Presentation/Demo/Examination at Tekniska museet is Sat 2026-05-23 11:00–16:00, **~1 day away**. Sign into Canvas in the controlled Chrome instance, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before the presentation to catch any last-minute room/time/format updates — there have been zero successful auto-checks in the entire run-up to this presentation. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-06-06 — Scheduled Canvas check FAILED (thirteenth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-06-03: logged-in Chrome at `/courses/59476/announcements` redirects to ADFS (login.ug.kth.se "Sign In" page); the public course-preview path (`?preview=1`) also redirects to ADFS; the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-06-30`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`. No new info captured. Notion project page updated with status comment. **Action for Gustav (URGENT — Inlämning 7 final report due in 4 days):** Inlämning 7 (Slutlig rapport) due Wed 2026-06-10 19:00. Sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before submitting to catch any last-minute updates. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track. Subsequent presentations (Tekniska 2026-05-23, KTH Innovation 2026-06-03) have both now passed.
- 2026-06-03 — Scheduled Canvas check FAILED (twelfth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-06-02: logged-in Chrome at `/courses/59476/announcements` redirects to ADFS (login.ug.kth.se "Sign In" page, confirmed via screenshot); the public course-preview path (`?preview=1`) also redirects to ADFS; the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-06-30`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`. No new info captured. Notion project page updated with status comment. **Action for Gustav (URGENT — KTH Innovation presentation TODAY 12:00–17:00):** do one manual check of https://canvas.kth.se/courses/59476/announcements before heading in to catch any last-minute room/format updates; sign into Canvas in the controlled Chrome instance so the next scheduled check can run. Inlämning 7 (final report) still due Wed 2026-06-10 19:00 (7 days away). Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-06-02 — Scheduled Canvas check FAILED (eleventh consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-06-01: logged-in Chrome at `/courses/59476/announcements` redirects to ADFS (login.ug.kth.se "Sign In" page, confirmed via screenshot); the public course-preview path (`?preview=1`) also redirects to ADFS; the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-06-30`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`. No new info captured. **Action for Gustav (URGENT — KTH Innovation presentation TOMORROW):** Presentation at KTH Innovation is Wed 2026-06-03 12:00–17:00, **1 day away**. Sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before tomorrow to catch any last-minute room/time/format updates for the KTH Innovation presentation — there have been zero successful auto-checks in the run-up. Inlämning 7 (final report) still due Wed 2026-06-10 19:00 (8 days away). Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-06-01 — Scheduled Canvas check FAILED (tenth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-05-29: logged-in Chrome at `/courses/59476/announcements` redirects to ADFS (login.ug.kth.se "Sign In" page); REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-06-30`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`; the public course-preview path that worked on 2026-05-14/15 (`?preview=1`) also redirects to ADFS. No new info captured. Notion **was** reachable this run — project page status comment posted. Next milestones: Presentation at KTH Innovation **Wed 2026-06-03 12:00** (2 days away) and Inlämning 7 (final report) **Wed 2026-06-10 19:00** (9 days away). **Action for Gustav (URGENT — KTH Innovation presentation in 2 days):** sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before Wed 3 Jun to catch any room/time/format updates for the KTH Innovation presentation. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-05-29 — Scheduled Canvas check FAILED (eighth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-05-28: logged-in Chrome at `/courses/59476/announcements` and `/courses/59476` both redirect to ADFS (login.ug.kth.se "Sign In" page); REST API endpoints (`/api/v1/announcements?context_codes[]=course_59476`, `/api/v1/courses/59476/discussion_topics?only_announcements=true`, `/api/v1/courses/59476/modules`, `/api/v1/courses/59476/assignments`) all return empty / unauthenticated; the public course-preview path that worked on 2026-05-14/15 remains closed. No new info captured. Notion **was** reachable this run — project page updated with a status comment. Next milestones: Presentation at KTH Innovation **Wed 2026-06-03 12:00** (5 days away) and Inlämning 7 (final report) **Wed 2026-06-10 19:00** (12 days away). **Action for Gustav:** sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before 3 Jun to catch any room/time/format updates for the KTH Innovation presentation. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-05-28 — Scheduled Canvas check FAILED (seventh consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-05-25: logged-in Chrome at `/courses/59476/announcements` and `/courses/59476` both redirect to ADFS (login.ug.kth.se "Sign In" page); the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-05-29`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}`; the public course-preview path that worked on 2026-05-14/15 remains closed. No new info captured. Notion **was** reachable this run — project page updated with a status comment. Next milestones: Presentation at KTH Innovation **Wed 2026-06-03 12:00** (6 days away) and Inlämning 7 (final report) **Wed 2026-06-10 19:00** (13 days away). **Action for Gustav:** sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before 3 Jun to catch any room/time/format updates for the KTH Innovation presentation. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
- 2026-06-26 — Scheduled Canvas check FAILED. Both access paths blocked: Claude-in-Chrome extension is not connected (no browser session reachable), and the sandbox HTTPS proxy returns `403 X-Proxy-Error: blocked-by-allowlist` for `canvas.kth.se` — the anonymous REST/preview path that worked on 2026-05-14/15 is unreachable from this scheduled run's network. No new info captured. **Note:** all standard-track deadlines have now PASSED — Inlämning 7 (2026-06-10), Inlämning 8 (2026-06-12), and Godkända rapporter (2026-06-17, 9 days ago). Remaining items are admin-only: Publiceringsmedgivande (individual sign + scan, send to course admin) and DIVA deposit (by administrator, not student). Notion project page updated with the same entry. **Actions for Gustav:** (1) start the Chrome extension / sign back into Canvas if you want auto-checks to resume — the course is effectively over for the standard track; (2) if Publiceringsmedgivande hasn't been submitted yet, that's the only outstanding student action; (3) consider disabling the canvas-check scheduled task now.
- 2026-06-29 — Scheduled Canvas check FAILED (same failure mode as 2026-06-26). Claude-in-Chrome extension not connected; sandbox proxy still returns `403 X-Proxy-Error: blocked-by-allowlist` for `canvas.kth.se`. No new info captured. All standard-track deadlines remain in the past (Inl. 7 / 8 / Godkända rapporter — Godkända was 12 days ago). Notion project page updated with status comment. **Action for Gustav:** the canvas-check scheduled task is now producing only no-op failure logs — recommend either (a) disabling it via Cowork's scheduled tasks UI, or (b) restarting Claude in Chrome and signing back into Canvas if you still want monitoring through e.g. the grading-publication window. Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-07 — Scheduled Canvas check FAILED. Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-08` returns empty body (effectively blocked/allowlist). Notion MCP requires re-auth this run — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 20 days ago). **Recommendation:** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and every check since 2026-05-15 (except two brief windows) has been a no-op failure. Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-08 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-09` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 21 days ago). **Strong recommendation (repeat):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs for the past ~6 weeks. Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-09 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07/08: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-10` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 22 days ago). **Strong recommendation (repeat):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May. Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-12 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-09: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-13` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 25 days ago). **Strong recommendation (repeat):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-13 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-12: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-14` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 26 days ago). **Strong recommendation (repeat, ~7 weeks running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-14 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-13: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-15` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 27 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-16 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-14: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-17` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 29 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-17 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-16: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-18` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 30 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-20 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-17: Claude-in-Chrome extension not connected ("Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-21` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 33 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-06-30 — Scheduled Canvas check FAILED (third consecutive failure since 2026-06-26). Claude-in-Chrome extension not connected (tabs_context_mcp returns "Claude in Chrome is not connected"); sandbox web_fetch of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476` returns empty body (effectively blocked). No new info captured. All standard-track deadlines remain in the past (Godkända rapporter was 13 days ago). Notion project page updated with status comment. **Action for Gustav:** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — it has produced only no-op failure logs since the course ended. Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-21 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-20: Claude-in-Chrome extension not connected (`mcp__claude-in-chrome` tools unavailable); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-22` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 34 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-22 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-21: Claude-in-Chrome extension not connected (`mcp__claude-in-chrome` tools return "Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-23` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 35 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-23 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-22: Claude-in-Chrome extension not connected (`mcp__claude-in-chrome` tools return "Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-24` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 36 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-27 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-23: Claude-in-Chrome extension not connected (`mcp__claude-in-chrome__tabs_context_mcp` returns "Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-28` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 40 days ago). **Strong recommendation (repeat, ~2 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-07-28 — Scheduled Canvas check FAILED. Same pattern as 2026-07-07 → 2026-07-27: Claude-in-Chrome extension not connected (`mcp__claude-in-chrome__tabs_context_mcp` returns "Chrome extension isn't reachable"); sandbox `web_fetch` of `https://canvas.kth.se/api/v1/announcements?context_codes[]=course_59476&start_date=2026-05-15&end_date=2026-07-29` returns empty body (blocked/allowlist). Notion MCP requires re-auth this run (`plugin:engineering:notion` unauthorized, per system reminder — needs re-authorization via claude.ai connector settings) — no project-page comment posted. All standard-track deadlines remain in the past (Godkända rapporter was 41 days ago). **Strong recommendation (repeat, ~2.5 months running):** disable the canvas-check scheduled task via Cowork's scheduled tasks UI — course is over for the standard track and this task has produced only no-op failure logs since mid-May (with brief exceptions). Only outstanding student action (if not already done): Publiceringsmedgivande — sign, scan as PDF `gustav-skarman`, email to `oviberg@kth.se`.
- 2026-05-25 — Scheduled Canvas check FAILED (sixth consecutive failure since 2026-05-15). Same pattern as 2026-05-18 → 2026-05-22: logged-in Chrome at `/courses/59476/announcements` and `/courses/59476` both redirect to ADFS (login.ug.kth.se "Sign In" page, confirmed via page text); the anonymous REST API (`/api/v1/announcements?context_codes[]=course_59476&start_date=2026-04-01&end_date=2026-05-26`) returns `{"status":"unauthenticated","errors":[{"message":"auktorisering av användare krävs"}]}` both via web_fetch (empty body) and in-browser; the public course-preview path that worked on 2026-05-14/15 remains closed. No new info captured. Notion **was** reachable this run — project page updated with a status comment. **Note:** the Tekniska museet presentation (Sat 2026-05-23) has now PASSED. Next milestones: Presentation at KTH Innovation **Wed 2026-06-03 12:00** (9 days away) and Inlämning 7 (final report) **Wed 2026-06-10 19:00** (16 days away). **Action for Gustav:** sign into Canvas in the controlled Chrome instance so the next scheduled check can run, AND do one manual check of https://canvas.kth.se/courses/59476/announcements before 3 Jun to catch any room/time/format updates for the KTH Innovation presentation. Last confirmed standard-track state (2026-05-15): latest announcement still 2026-03-06 "Biblioteket erbjuder Workshops"; nothing newer on the standard track.
