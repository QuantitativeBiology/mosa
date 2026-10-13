# Documentation site

[Documentation](../index.md)

The documentation content, `mkdocs.yml`, `requirements-docs.txt`, and custom theme (`theme/`) live under `docs/`. The build excludes these tooling files from the published site. The site is built using [MkDocs](https://www.mkdocs.org/) and the repository's custom theme. [GitHub](https://github.com/) continues to render the same Markdown files individually; [GitHub Pages](https://pages.github.com/) serves the website with its sidebar, search, and page navigation.

## Preview locally

Run these commands from the repository root to install the documentation tools separately from the library:

```bash
python -m venv /tmp/mosa-docs-env
source /tmp/mosa-docs-env/bin/activate
python -m pip install -r docs/requirements-docs.txt
mkdocs serve -f docs/mkdocs.yml
```

Open the address printed by [MkDocs](https://www.mkdocs.org/), usually `http://127.0.0.1:8000/mosa/`. Changes to Markdown, navigation, and theme files rebuild the preview automatically. Restart the server after editing `docs/theme/hooks.py`.

```bash
mkdocs build --strict -f docs/mkdocs.yml
```

A strict build checks the configuration, internal links, and heading anchors, then writes the static site to `site/`.

## Publish on GitHub Pages

The documentation workflow builds on pull requests and branch pushes. It deploys only from `main`, including a manual workflow run on that branch. In the repository's **Settings → Pages**, select **[GitHub Actions](https://docs.github.com/en/actions)** as the source before the first deployment.

After publication, readers can use the README's Documentation link or visit [quantitativebiology.github.io/mosa](https://quantitativebiology.github.io/mosa/). No [Python](https://www.python.org/) installation is needed to read the site. [GitHub Pages](https://pages.github.com/) hosting is free for public repositories.

The workflow uploads the generated site as a Pages artifact rather than committing generated HTML. The README remains the project introduction; the website provides the guides and reference pages.

## Edit content and navigation

Edit Markdown under `docs/` and update `nav` in `docs/mkdocs.yml` when adding or moving a page. Link command names in prose to their command-reference pages; keep commands inside code blocks as plain text for copying. Link external project names to their official documentation or project website. Use relative links so pages remain readable on [GitHub](https://github.com/); a build hook converts links to repository code and root files into [GitHub](https://github.com/) URLs.

Explain operations and constraints directly, keeping defaults in reference tables and procedures in guides. Specify which behaviour belongs to the library or a particular model, and keep important restrictions beside the relevant operation. Remove repeated introductions and commentary that adds no technical information.


## Theme

`docs/theme/main.html` defines the page structure; `css/site.css` and `js/site.js` define styling, mobile navigation, and search. Fonts use the system sans-serif stack for prose and Menlo with monospace fallbacks for code and navigation labels. Search uses the locally generated [MkDocs](https://www.mkdocs.org/) index without an external service.

See [MkDocs configuration](https://www.mkdocs.org/user-guide/configuration/) and [GitHub Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for the build and hosting settings.

## Check a documentation change

Keep each explanation in one place: task guides own procedures, model pages own model behaviour, and reference tables own exact fields and defaults. Link to the relevant section when readers need more detail, rather than repeating a summary on several overview pages.

After moving a page, update navigation and incoming links, then run the strict build. When changing a command example, execute it from the stated working directory. The first-run fixtures live in `docs/examples/`; their small training budget keeps the complete conversion, training, and projection sequence practical to check locally.

For changes to the theme, check desktop and narrow-screen navigation, search, code blocks, and diagrams.
