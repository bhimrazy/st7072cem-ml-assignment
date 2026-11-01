# Paper

The coursework paper in the IEEE conference template (`IEEEtran`), with a 6,000-word budget.

| File | What it is |
|---|---|
| `main.tex` | Paper source; `main.pdf` is the compiled paper |
| `references.bib` | Reference list (76 entries); BibTeX prints only the cited ones |
| `tables/` | LaTeX tables generated from `../results/` by `uv run python -m src.tables` |
| `figures/` | Figures (vector PDF and 300 dpi PNG), written by the notebooks in `../notebooks/` |

Build with pdfLaTeX, as the IEEE template intends (TeX Live or TinyTeX with the `ieeetran`
package):

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or upload this folder to Overleaf and compile with pdfLaTeX.

The word count (abstract, headings and body text from the Introduction to the end of the
Conclusion) excludes captions, references and appendices.
