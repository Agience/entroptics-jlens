# Contributing to Entroptics JLens

Entroptics reads of the Jacobian lens. **entroptics is the engine, not a dependency.**

## Rule zero: check entroptics first

    python -c "import entroptics, pathlib; print(pathlib.Path(entroptics.__file__).parent)"

Read it **before** writing any spectral, statistical or scaling primitive. If a library read does
not fit, **measure** that and show the number. The library is
[`entroptics`](https://pypi.org/project/entroptics/) on PyPI.

## Tests

```bash
python -m pytest -q
python -m ruff check src tests research/experiments
```

Run both after editing prose as well as code: [`tests/test_docs.py`](tests/test_docs.py) checks
every markdown file for literal tabs, stray carriage returns, unbalanced math, ragged tables and
dead links.

## Rules

1. Never reimplement a floor, a rank, a whiten or an occupancy.
2. **No tuned constants in the decision path.** Anything typed names the error or geometry it
   comes from.
3. **Measure, do not fit.** Nothing here trains.
4. **Report what you measured.** A measurement that does not support the hypothesis is still the
   measurement.
5. Every number quoted anywhere has a line in [`research/PAPER.md`](research/PAPER.md) saying what
   it was measured on, changed in the same commit.

## Nothing truncates or substitutes silently

A missing lens, a non-square transport, an unknown null provider: each **refuses**, naming what it
found. A subset of layers is always named explicitly — the CLI's `--layers` defaults to `all` and
prints which layers it read, and the experiment scripts require it outright.

## Releasing

1. **Finish `main` first.** Suite green, `python -m build`, `python -m twine check dist/*`.
2. **Then tag.** `gh release create vX.Y.Z --target main`.
3. **Use a version number once.** If anything changes after a release, bump to the next patch.

[`.github/workflows/publish.yml`](.github/workflows/publish.yml) runs on the release: it builds,
tests the sdist it built, and uploads through PyPI Trusted Publishing.

Cite the **concept DOI**. It follows the newest deposit; a version DOI pins one archive.

## Contributing

Fork, branch from `main`, sign off every commit (`git commit -s`) to certify the
[DCO](https://developercertificate.org/), open a PR. Commit format: `fix:` / `feat(scope):` /
`docs:` / `test:` / `chore:`. By contributing you agree your contribution is Apache-2.0 (per section
5), including its section 3 patent grant.

**Security vulnerabilities: do not open a public issue** — email **connect@agience.ai**.

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
