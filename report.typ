// ============================================================
//  MMA3001 - Numerical Methods and Machine Learning
//  Individual Project Report
// ============================================================

// ---------- Base formatting ----------
#set text(font: "New Computer Modern")
#show math.equation: set text(font: "New Computer Modern Math")
#set page(numbering: "1")
#set heading(numbering: "1.")

// ---------- Figure width constants ----------
// Use `width: plotWidth` for figures in the normal single-column flow,
// and `width: twoColPlotWidth` for figures inside a #columns(2)[...] block.
#let plotWidth = 70%
#let twoColPlotWidth = 80%

// ---------- Drafting helper ----------
// Marks guidance still to be replaced with real content.
// Delete this definition and every #todo[...] call before submitting.
#let todo(body) = block(
  fill: luma(94%),
  stroke: (left: 2pt + luma(60%)),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
  text(size: 0.9em, style: "italic", body),
)

// ---------- Document metadata ----------
#set document(
  title: "Automated Detection of Pork Rasher Packaging Errors and Meat Quality",
  author: "Alec Stanley",
)

// ============================================================
//  Title page
// ============================================================
#show title: set align(center)
#title()
#align(center)[
  Alec Stanley

  #v(0.5em)

  MMA3001 --- Numerical Methods and Machine Learning

  Dataset 3 --- Frozen-Meat Packaging Quality
]

#pagebreak()

// ============================================================
//  Table of contents
// ============================================================
#outline()

#pagebreak()

// ============================================================
//  Body
// ============================================================

= The Engineering Problem

Need to cite @pork-rasher-error-packaging_dataset

#todo[
  Explain: the engineering context; the problem being addressed; why the problem
  matters; the intended use of the computational solution; and the limitations of
  the selected scope.
]

= Inputs, Outputs, and Data

#todo[
  Define: the input data accepted by the solution; the expected format, units and
  domain of the inputs; the outputs produced; the meaning and intended use of each
  output; and how invalid, missing or unsupported inputs are handled.

  Address top-level inputs and outputs first, then the specific inputs and outputs
  of the functions you developed (docstrings and a documentation generator can
  assist here).
]

= The Computational Approach

#todo[
  Explain: how the method works at an appropriate level; why it was selected; its
  important assumptions; its settings or parameters; how it was implemented; and
  its known limitations.
]

= The Validation Method and Results

#todo[
  Explain: what was validated; why the validation method is appropriate; which
  metrics or acceptance criteria were used; whether the validation information
  influenced model selection; and what the validation cannot establish.

  A numerical result or a visually convincing output is not sufficient evidence of
  accuracy.
]

= Comparison and Optimisation Work

#todo[
  Identify and compare at least one credible alternative or baseline. The
  comparison may address accuracy, robustness, computational time, memory use,
  implementation complexity, interpretability, scalability, or suitability for the
  intended engineering application. The purpose is to demonstrate the final
  approach was selected using evidence rather than preference alone.

  Then analyse and, where appropriate, improve performance. Make use of profilers
  and timing routines, and where appropriate estimates of arithmetic intensity and
  the number of FLOPs, to explain and justify the performance of your code.
  Compare the baseline solution with the improved or selected solution and justify
  the final design.
]

// Suggested comparison table --- populate with real measurements.
#figure(
  table(
    columns: 4,
    align: (left, center, center, center),
    table.header([*Method*], [*Accuracy*], [*Runtime*], [*Memory*]),
    [Baseline], [--], [--], [--],
    [Alternative], [--], [--], [--],
    [Final], [--], [--], [--],
  ),
  caption: [Comparison of the baseline, alternative and final approaches.],
) <tab-comparison>

= Limitations and Lessons Learned

#todo[
  An unsuccessful or partially successful method can still demonstrate excellent
  engineering work if you validate it honestly, identify its limitations, explain
  what was learned, and propose well-justified improvements.
]

= AI-use disclosure

#todo[
  Disclose all material AI use. Explain: which AI tools were used; what they were
  used for; the approximate level of AI contribution; why AI was used for those
  tasks; how AI-generated material was checked; which important decisions remained
  your responsibility; any errors, limitations or unhelpful suggestions produced by
  AI; and how AI use affected your understanding or workflow.

  Disclosure does not replace verification --- you remain responsible for all
  submitted code, results, claims, references and engineering decisions.
]

#bibliography(
  "references.bib",
  style: "apa",
  title: "References",
)

#pagebreak()

= Appendix
