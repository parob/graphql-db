# GraphQL-DB

> Compatibility note: `AGENTS.md` and `CLAUDE.md` are both supported in this repo.
> Keep these files identical. Any change in one must be mirrored in the other.

SQLAlchemy integration for graphql-api — auto-generates GraphQL types from ORM models. Published on [PyPI](https://pypi.org/project/graphql-db/).

## Project Structure

| Directory | Description |
|-----------|-------------|
| `graphql_db/` | Main package source |
| `tests/` | Test suite (13 files, pytest) |
| `docs/` | Documentation |

## Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check graphql_db tests
```

Note: This library uses `hatchling` as its build backend and stores the version directly in `pyproject.toml` (unlike the other graphql-* libraries which use `setuptools-scm` with a `VERSION` file).

## Key Patterns

- `ModelBase` provides `id` (UUID), `query()`, `filter()`, `get()`, `create()`, `delete()`
- `DatabaseManager` handles engine/session lifecycle; `with_db_session()` wraps execution with auto-commit/rollback
- `GraphQLSQLAlchemyMixin` converts columns, relationships, and hybrid properties into GraphQL fields automatically
- Relay pagination via `relay_connection(Model)` → cursor-based `edges`/`pageInfo`

## Releasing

See the ecosystem-level `CLAUDE.md` in the parent workspace for the full release process. In short:

```bash
# Ensure CI is green on main, then:
git tag X.Y.Z
git push origin X.Y.Z
```

CI publishes to PyPI and creates a GitHub Release automatically. Note: the publish workflow updates the version in `pyproject.toml` directly (not a VERSION file).
