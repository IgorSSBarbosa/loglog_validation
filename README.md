# loglog_validation

Simulation validation for the paper **"The Log-Log Plot Technique"**
(`../article_writting/article.tex`). See `PLAN.md` for the full plan, the article's
"paper objects" cross-reference table, ground rules, and the experiment ladder;
`TODO.md` for current status.

Start here: `experiments/00_synthetic/README.md`.

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned to Python 3.9.2 / numpy 2.0.2 / scipy 1.13.1 / matplotlib 3.9.4 / pytest 8.4.2 —
same versions as `../loglog_experiments/`'s environment. `.venv/` is gitignored; recreate
it with the commands above rather than committing it.

This is a from-scratch restart of `../loglog_experiments/` (kept as historical
reference, not reused) — see `PLAN.md` for why.
