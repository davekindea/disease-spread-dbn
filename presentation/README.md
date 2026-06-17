# Project presentation (LaTeX Beamer)

## Files

- `PGM_Project_Presentation.tex` — formal slide deck

## Compile

Requires a LaTeX distribution with Beamer (TeX Live, MiKTeX).

```bash
cd presentation
pdflatex PGM_Project_Presentation.tex
pdflatex PGM_Project_Presentation.tex
```

Output: `PGM_Project_Presentation.pdf`

## Before presenting

1. Replace `Your Name` in the `.tex` file (line with `\author{...}`).
2. Run the notebook once so figures exist in `../outputs/notebook/`.
3. Re-run the results cell if you change `max_nodes` or EM iterations — update the numbers on the **Dataset summary** and **Learned parameters** slides if needed.

## Figures used

Slides pull PNGs from `outputs/notebook/` (fig1a through fig6).
