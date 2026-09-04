# Contributing

Use test-driven changes: add or update the smallest failing test, implement the
change, then run the relevant integration and paper regressions when scientific
behavior can change.

```bash
ruff format .
ruff check .
mypy src tests
pytest -m "not slow and not remote" -q
ULTR_RUN_HF_TESTS=1 pytest tests/integration/test_hf_release.py -q
ULTR_RUN_PAPER_TESTS=1 pytest tests/paper -q
```

Do not add Lightning `.ckpt`, NetCDF, Parquet, or generated prediction files to
this repository. Binary research assets belong in the appropriate Hugging Face
repository.
Never commit access tokens or W&B credentials.
