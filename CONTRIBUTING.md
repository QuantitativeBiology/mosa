Contributing to MOSA

Contributions can include bug fixes, clearer documentation, tests, and new integration methods.

## Report a problem or propose a change

Open a [GitHub issue](https://github.com/QuantitativeBiology/mosa/issues) with a clear description. For bugs, include:

- The command you ran, the expected behavior, and the actual output.
- The MOSA revision (`git rev-parse HEAD`), [Python](https://www.python.org/) version, operating system, and accelerator, if relevant.
- A minimal configuration and small synthetic example that reproduce the problem, where possible.

Running a command with `--debug` provides more diagnostic information. Remove credentials, private paths, and sensitive sample information before sharing logs or data.

Discuss substantial changes in an issue before implementation so the scope and interface can be agreed on.

## Development setup

Fork the repository on [GitHub](https://github.com/), clone your fork, and create a branch for your change. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,lint,mofa]"
```

On Windows, use `.venv\Scripts\activate` to activate the environment. The `mofa` extra installs the optional backend used by the full test suite.

## Check your changes

Run the same core checks used by CI:

```bash
ruff check src tests .github/scripts
ruff format --check src tests .github/scripts
python -m unittest discover -s .github/scripts -p 'test_*.py' -v
mypy
pytest -q -m "not behavioral"
```

If your change affects learning behavior, also run `pytest -q` to include the behavioral tests. For documentation changes, install `docs/requirements-docs.txt` and run `mkdocs build --strict -f docs/mkdocs.yml`. Use `mkdocs serve -f docs/mkdocs.yml` to inspect the site; see [Documentation site](docs/development/documentation-site.md) for preview and publication instructions.

Keep changes focused and add or update tests when behavior changes. Update the relevant documentation when changing a command or configuration option.

## Adding integration methods

Follow [Adding a model](docs/development/adding-models.md) for configuration, implementation, and registration. The [model contract](docs/reference/model-interface.md) explains the data, persistence, output, and evaluation requirements; the [code architecture](docs/development/architecture.md) shows where these operations live.

Document the method's supported operations and restrictions alongside its configuration. Test the lifecycle through the shared registry, including held-out reconstruction when supported, rather than only calling the implementation directly.

## Submit a pull request

Explain the problem, what changes for users, and how you verified the result. Link related issues and include example output or screenshots where they help reviewers assess the change.

## Publish a release

The `.github/workflows/pypi-publish.yaml` workflow publishes `mosalib` to PyPI
using Trusted Publishing.

Set the same stable `X.Y.Z` version in `pyproject.toml` and
`src/mosa/__init__.py`, then merge the change and release workflow into `main`.
For example, for the `0.1.0` release:

```bash
git switch main      # use main as the branch to tag
git pull --ff-only   # update
git tag v0.1.0       # tag the branch. must start with 'v'
git push origin v0.1.0 # upon push, the tagged version will be puslished to PyPI
```

Pushing the tag triggers source/version checks, the existing CI tests, lint and
type checks, then a build. The workflow checks both distribution versions and
metadata, installs the wheel in a clean environment, checks dependencies, and
smoke tests the import and CLI outside the checkout. Only those validated
artifacts are uploaded to PyPI and attached to a GitHub Release. Behavioral
tests retain their nightly schedule and do not gate releases.

Release tags must use `vX.Y.Z`, match both source versions, and point to a commit
in `main`. A manual run on a branch validates and builds without publishing;
a manual run on a release tag can publish. PyPI does not allow overwriting
previously uploaded files: use a new version for a correction. If only the
GitHub Release job fails after upload, rerun that failed job rather than the
successful publishing job. GitHub Actions used by the release workflow and its
reused CI are pinned to commit SHAs.

Return to the [README](README.md) or browse the [documentation](docs/index.md).

