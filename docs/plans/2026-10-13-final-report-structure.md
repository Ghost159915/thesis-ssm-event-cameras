# Final report structure (decided 2026-10-13)

**Files (2026-10-13):** the full 64-page draft was renamed `thesis/latex/draft_main.tex` (source material, all
measured numbers, `\res{}` placeholders). The new `thesis/latex/main.tex` is the report the author writes: the
structure below with headings, bullet points of what each section should cover, and every equation, table and figure
from the draft placed in its new section (no prose); Chapter 2 (Literature Review) and the nomenclature are copied
verbatim. Front matter, roman/arabic/roman numbering and running header are in place; it builds with 0 errors and 0
undefined references (52 pages; the body is 32 pages of headings, bullets and floats).
`conference.tex` was deleted. zID z5457896 is on the title page; degree name and submission date are `\res{}`
placeholders.

**Decision (user, after discussing with supervisor Will Midgley on 2026-10-13):** submit the **standard thesis
report**, about **50 pages of body, not counting the bibliography and appendices**, instead of a ~20-page conference
paper. This supersedes Part B of `2026-10-11-post-training-spiking-analysis.md`. The 10-minute presentation is
unchanged. `thesis/latex/conference.tex` (headings-only skeleton) is no longer needed.

**Reference report:** Charles Sargent, *Development of a Buoyancy Control System for the BlueROV 2 Underwater
Remotely Operated Vehicle*, BE (Mechatronic) thesis, UNSW School of Mechanical and Manufacturing Engineering,
submitted 28 April 2025, supervisor Will Midgley. 97 PDF pages, US Letter, Microsoft Word. The PDF lives outside the
repo (`~/Downloads/Charles Sargent Thesis Report.pdf`). The structure below is taken from it as a guide, not a
template to copy.

---

## 1. Structure of the reference report

### Front matter (lower-case roman page numbers, i–xi)

| # | Page | Element | Notes |
|---|---|---|---|
| 1 | i | Title page | School; Faculty; UNSW Sydney; "by" + name; title; "Thesis submitted as a requirement for the degree of Bachelor of Engineering in …"; submitted date; zID; supervisor |
| 2 | ii | Originality statement | standard UNSW declaration, signed and dated |
| 3 | iii | Abstract | one paragraph, ~150 words: context → gap → what was done → headline number |
| 4 | iv | Acknowledgements | in the front matter, before the contents |
| 5 | v–vii | Table of contents | three heading levels (e.g. 2.1.3) |
| 6 | viii–ix | List of figures | ~78 figures in total (incl. appendices) |
| 7 | x | List of tables | ~20 tables |
| 8 | xi | Nomenclature | abbreviations, alphabetical |

### Body (arabic page numbers 1–50; exactly 50 pages)

| Ch. | Title | Pages | Length | Words | Sections |
|---|---|---|---|---|---|
| 1 | Introduction | 1–2 | 2 pp | ~625 | 1.1 Importance of the problem; 1.2 Context of this work (ends with a roadmap paragraph: "Section 2 … Section 7") |
| 2 | Literature Review | 3–17 | 15 pp | ~5,640 | 2.1 the technique families (8 subsections, one per method, each critically compared); 2.2 control methods (4 subsections) |
| 3 | Research Question | 18 | 1 pp | ~370 | 3.1 Research question and aims (one-sentence question in bold-ish standalone line, then 5 numbered, measurable aims, e.g. "model with ±10 % error"); 3.2 Hypothesis |
| 4 | Methodology | 19–35 | 17 pp | ~5,980 | 4.1 Hardware; 4.2 Software; 4.3 Modelling; 4.4 Simulation; 4.5 Controllers; 4.6 Testing procedure (experimental + simulation) |
| 5 | Results | 36–46 | 11 pp | ~3,470 | one section per experiment (5.1–5.7), each figure + table, then 5.8 Summary of results (summary table) |
| 6 | Discussion | 47–49 | 3 pp | ~1,150 | 6.1–6.3 each tied explicitly to the numbered aims and the hypothesis ("fulfils aim 1 of section 3.1"); 6.4 Future research opportunities (numbered list, then a paragraph each) |
| 7 | Conclusion | 50 | 1 pp | ~450 | one page: hypothesis confirmed? research question answered? then a recap of lit review → method → results → future work |
| | **Body total** | 1–50 | **50 pp** | **~17,700** | |

### Back matter (roman numbering continues from the front matter: xii onwards)

| Element | Pages | Notes |
|---|---|---|
| Bibliography | xii–xiv | IEEE numeric, 57 references, in order of first citation |
| Appendices A–L | xv–xlvii | each lettered with its own heading: calculations, technical drawings, manufacturing, code listings, procedures |

### Conventions worth copying

1. **The page count is enforced by the numbering:** front and back matter are roman, only the body is arabic, so
   "50 pages" is literally pages 1–50.
2. **Running header** on every page: author name + short title.
3. **Research question gets its own short chapter** between the literature review and the method: one question,
   numbered measurable aims, an explicit hypothesis. Discussion then answers aim by aim, and the conclusion says
   whether the hypothesis held.
4. **Every figure and table is numbered and captioned**, and listed in the List of Figures / List of Tables.
5. **Results end with a summary section and summary table**; Discussion interprets, it does not repeat numbers.
6. **Future work sits inside the Discussion** (6.4), so the Conclusion stays one page.
7. **Bibliography comes before the appendices.**
8. Proportions: literature review 30 % of the body, methodology 34 %, results 22 %, discussion + conclusion 9 %,
   introduction + research question 5 %.

---

## 2. The draft (`draft_main.tex`) against the reference

State of the draft (build of 2026-10-11, then `thesis/latex/main.tex`, now `draft_main.tex`, `report` class, 11 pt, A4, 1-inch margins, IEEE
`ieeetr`): 64 pages; body Ch. 1–7 = pages 6–51 (**46 pages, ~19,600 words**).

| Reference (Sargent) | pp | Ours now | pp | Words | Action |
|---|---|---|---|---|---|
| Title page | i | title page | 1 | | add degree statement, School/Faculty, zID, submitted date, supervisor |
| Originality statement | ii | — | | | **add** (UNSW standard declaration) |
| Abstract | iii | Abstract | 2 | 396 | fine; move acknowledgements to follow it |
| Acknowledgements | iv | after Ch. 7 | 52 | 41 | **move to front matter** |
| Contents | v–vii | Contents | 3–4 | | fine |
| List of figures / tables | viii–x | — | | | **add** `\listoffigures`, `\listoftables` |
| Nomenclature | xi | Nomenclature | 5 | | fine |
| 1 Introduction | 2 pp | Ch. 1 Introduction (incl. Research Question, Contributions, Scope) | 3 pp | 1,057 | move Research Question into a new Ch. 3; keep contributions and scope |
| 2 Literature Review | 15 pp | Ch. 2 Literature Review | 15 pp | 7,059 | same length; trim the Thesis-A UAV-era sections if space is needed |
| 3 Research Question | 1 pp | — (it is §1.1 + the pre-registration) | | | **new short chapter**: question, numbered measurable aims, hypotheses (receptive field → AP_L; rate robustness without Δt compensation; ladder ordering analog ≥ graded > binary) |
| 4 Methodology | 17 pp | Ch. 3 Methodology + Ch. 4 Experimental Setup | 6 + 4 pp | 2,183 + 1,654 | keep as two chapters or merge; either way ~10 pp |
| 5 Results | 11 pp | Ch. 5 Results (§5.7 spiking still to fill) | 9 pp | 3,460 | add §5.7 numbers/figures (+2–3 pp) and a closing **Summary of results** table |
| 6 Discussion | 3 pp | Ch. 6 Discussion (has aims table §6.3) | 5 pp | 2,325 | tie each section to the numbered aims of the new Ch. 3 |
| 7 Conclusion | 1 pp | Ch. 7 Conclusion and Future Work | 4 pp | 1,852 | optionally move Future Work into Discussion so the conclusion is ~1–2 pp |
| Bibliography, then Appendices | xii– | Appendices A–D, then Bibliography | 53–64 | | **swap order**: bibliography first |
| roman back matter | | arabic throughout | | | **number front + back matter in roman**, body 1–N arabic |
| running header | | none | | | add name + short title (`fancyhdr`) |

**Length:** the body is already 46 pages. With the new Research Question chapter (~1 p), §5.7 results (+2–3 pp) and a
results summary (~1 p), it lands at roughly **50 pages**, so the target is met by structure, not by cutting.
If it runs over, trim the literature review first (7,059 words vs the reference's 5,640).

---

## 3. Open questions (check before restructuring)

- [ ] **Exact length rule** for Thesis C (MMAN4953): is it 50 pages of body, a word limit, or both? Font/spacing rules?
      Only the MMAN4951 rubric is filed (`thesis/admin/`). Get the MMAN4953 rubric.
- [x] **zID**: z5457896 (on the title page).
- [ ] Degree name for the title page (e.g. "Bachelor of Engineering in Mechatronic Engineering").
- [x] Methodology and Experimental Setup: **merged** into one Chapter 4 (user, 2026-10-13).
- [ ] Future Work: inside Discussion (reference) or its own section of Ch. 7 (ours now)?
- [x] `conference.tex`: **deleted** (user, 2026-10-13).
