## Summary

<!-- What does this change do, and why? -->

## Documentation sync

For any user-facing change (new or changed behavior, API, installation, defaults):

- [ ] `CHANGELOG.md` has an entry under `Unreleased` in this pull request.
- [ ] `README.rst` is updated where it describes the change.
- [ ] The linked pages under `docs/` are updated.
- [ ] Code snippets in the README and docs still run as written.
- [ ] Not user-facing: none of the above applies.

## Checks

- [ ] Tests pass, and the core package still works without optional extras.
- [ ] The docs build in strict mode (`sphinx-build -W`).
- [ ] Commit messages and this description contain no private or unpublished
      information.

Release steps are described in [RELEASING.md](../RELEASING.md).
