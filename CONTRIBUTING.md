# Contributing to MOSA

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
ruff check src tests
ruff format --check src tests
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

Return to the [README](README.md) or browse the [documentation](docs/index.md).
