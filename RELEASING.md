# Releasing

This page is the maintainer checklist for cutting a release of skbel. It also
holds the text of a publishing workflow, which is added to the repository as a
separate step.

## When a release is complete

A release is complete when all three of the following are true:

1. The version is on PyPI.
2. The exact Zenodo version DOI for that release is recorded in the README
   citation and in `CITATION.cff`.
3. The Read the Docs `stable` build shows that version.

A Git tag or a GitHub release alone is not a complete release. State which of
the three items are done and which are pending until all can be checked.

## Versioning

Versions follow [semantic versioning](https://semver.org/):

- Patch (`x.y.Z`): bug fixes that keep the documented behavior.
- Minor (`x.Y.0`): new features, or a change in numerical output that is
  described under "Changed behavior" in the changelog.
- Major (`X.0.0`): incompatible changes to the public API.

Never rewrite a published tag, and never re-upload a different artifact under a
version that has already been published.

## Keeping user-facing documentation in sync

Every pull request that changes user-facing behavior updates, in the same pull
request:

- the `Unreleased` section of `CHANGELOG.md`;
- the README, including installation instructions and code snippets;
- the linked pages under `docs/`.

Snippets in the README and in the docs must run as written, and the test suite
should execute them where practical. The pull request template carries the same
checklist.

## Before tagging

1. Move the `Unreleased` entries of `CHANGELOG.md` under a heading for the new
   version and date.
2. Set the same version in `pyproject.toml`, `CITATION.cff` and the changelog
   heading.
3. Check that the core package depends only on NumPy and the scientific Python
   stack declared in `pyproject.toml`, and that importing `skbel` works without
   the optional extras. Optional extras (such as `neural` and `bnn`) must stay
   optional, and tests that need them skip cleanly when they are missing.
4. Run the test suite and build the documentation in strict mode, so that any
   warning fails the build:

   ```sh
   uv run pytest
   uv run sphinx-build -W -b html docs docs/_build/html
   ```

5. Build the source distribution and the wheel and inspect them:

   ```sh
   uv build
   ```

   - List the contents of both artifacts. The wheel should contain the package
     files and the license, and not the `.npy` reference arrays under
     `skbel/testing/`. The source distribution should contain the changelog, the
     citation file, the sources and the seven source test fixtures that the test
     suite needs.
   - Confirm that neither artifact includes local paths, credentials, private
     notes, unpublished data or personal addresses, and no `.DS_Store` files.
   - Confirm that the metadata (version, Python requirement, extras, URLs) is
     correct.

6. If a release archive (tarball) is built or uploaded by hand, create it with
   numeric owner and group set to 0 (`--owner=0 --group=0 --numeric-owner` for
   GNU tar, `--uid 0 --gid 0` for bsdtar) and without `.DS_Store` files.
7. Check that the package metadata, commit messages, pull request text and the
   changelog are accurate release notes and carry no personal contact details.
   Commits use the GitHub `users.noreply.github.com` address.

## Publishing to PyPI with a trusted publisher

PyPI supports publishing from GitHub Actions without a long-lived token, using
OpenID Connect. See the PyPI documentation on
[adding a trusted publisher](https://docs.pypi.org/trusted-publishers/adding-a-publisher/)
and [using a trusted publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

Setup:

- register the trusted publisher for the project on PyPI, using the repository,
  the workflow file name and the environment name that appear in the workflow
  below;
- create the matching `pypi` environment in the GitHub repository settings;
- save the workflow below as `.github/workflows/publish.yml`.

The repository itself holds no publisher configuration, and no credentials are
needed.

The workflow text is kept here rather than under `.github/workflows/`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - name: Build the distributions
        run: |
          python -m pip install --upgrade build
          python -m build
      - uses: actions/upload-artifact@v4
        with:
          name: python-package-distributions
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/project/skbel/
    permissions:
      id-token: write
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: python-package-distributions
          path: dist/
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
```

The build job has read-only access and no publishing permission. Only the
publish job can request the identity token, and it runs in the `pypi`
environment, where a manual approval can be required.

## After publishing

1. Wait for the archive of the release on Zenodo and copy its exact version DOI
   into both the README citation and `CITATION.cff`. If a citation pull request
   is already open, add it there; otherwise open a pull request for it.
2. Confirm that the Read the Docs `stable` build shows the new version.
3. Confirm that `pip install skbel==<version>` installs the published release
   in a clean environment.
