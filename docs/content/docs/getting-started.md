---
title: "Getting Started"
weight: 1
---

# Getting Started with GraphQL DB

GraphQL DB provides SQLAlchemy integration for [graphql-api](https://graphql-api.parob.com/), enabling automatic GraphQL schema generation from database models.

## Installation

Install GraphQL DB with SQLAlchemy:

```bash
pip install graphql-db graphql-api sqlalchemy
```

Or using UV (recommended):

```bash
uv add graphql-db graphql-api sqlalchemy
```

## Prerequisites

GraphQL DB extends graphql-api. If you're new to building GraphQL APIs in Python, first check out the [graphql-api getting started guide](https://graphql-api.parob.com/docs/fundamentals/getting-started/).

## Your First Database API

### 1. Set Up the Database

```python
from graphql_db.orm_base import DatabaseManager, ModelBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

# Initialize database (SQLite for this example)
db_manager = DatabaseManager(url="sqlite:///myapp.db")
```

### 2. Define Your Models

```python
class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)

# Create tables
db_manager.create_all()
```

### 3. Create GraphQL API

```python
from graphql_api import GraphQLAPI

api = GraphQLAPI()

@api.type(is_root_type=True)
class Query:
    @api.field
    def users(self) -> list[User]:
        """Get all users."""
        return User.query().all()

    @api.field
    def user(self, id: str) -> User | None:
        """Get a user by ID."""
        import uuid
        return User.get(uuid.UUID(id))
```

> **Note**: For more on defining GraphQL schemas, see the [graphql-api documentation](https://graphql-api.parob.com/).

### 4. Execute Queries

```python
# Define a query function
def run_query():
    result = api.execute('''
        query {
            users {
                id
                name
                email
            }
        }
    ''')
    return result

# Execute with automatic session management
result = db_manager.with_db_session(run_query)()
print(result.data)
```

## Adding Mutations

Create and modify database records:

```python
@api.type(is_root_type=True)
class Root:
    @api.field
    def users(self) -> list[User]:
        return User.query().all()

    @api.field(mutable=True)
    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(name=name, email=email)
        user.create()  # Save to database
        return user

    @api.field(mutable=True)
    def update_user(self, id: str, name: str | None = None) -> User | None:
        """Update a user."""
        import uuid
        user = User.get(uuid.UUID(id))
        if user and name:
            user.name = name
            user.create()  # Update in database
        return user

    @api.field(mutable=True)
    def delete_user(self, id: str) -> bool:
        """Delete a user."""
        import uuid
        user = User.get(uuid.UUID(id))
        if user:
            user.delete()
            return True
        return False
```

Execute a mutation:

```python
def run_mutation():
    return api.execute('''
        mutation {
            createUser(name: "Alice", email: "alice@example.com") {
                id
                name
                email
            }
        }
    ''')

result = db_manager.with_db_session(run_mutation)()
```

## Session Management

GraphQL DB uses context managers for automatic session handling:

```python
# Automatic session management
result = db_manager.with_db_session(lambda: api.execute(query))()

# For multiple operations
def complex_operation():
    # Create user
    user = User(name="Bob", email="bob@example.com")
    user.create()

    # Query all users
    all_users = User.query().all()

    # Execute GraphQL query
    result = api.execute('{ users { name } }')
    return result

# Session managed automatically
result = db_manager.with_db_session(complex_operation)()
```

## Database Configuration

### SQLite (Development)

```python
db_manager = DatabaseManager(url="sqlite:///development.db")
```

### PostgreSQL (Production)

```python
db_manager = DatabaseManager(
    url="postgresql://user:password@localhost/mydb"
)
```

### Environment-Based

```python
import os

db_manager = DatabaseManager(
    url=os.getenv('DATABASE_URL', 'sqlite:///default.db')
)
```

## Next Steps

Now that you have a basic API running:

- **[Models & Schema](models-schema/)** - Learn about model types and relationships
- **[Pagination](pagination/)** - Implement cursor-based pagination
- **[Performance](performance/)** - Optimize queries for production
- **[Examples](examples/)** - See complete examples

## Common Patterns

### With HTTP Server

Serve your API over HTTP using [graphql-http](https://graphql-http.parob.com/):

```python
from graphql_http import GraphQLHTTP

server = GraphQLHTTP.from_api(api)

# Wrap with session management
@server.app.middleware("http")
async def add_db_session(request, call_next):
    def handler():
        return call_next(request)
    return db_manager.with_db_session(handler)()

server.run()
```

### Testing

Use in-memory databases for testing:

```python
import pytest

@pytest.fixture
def db():
    db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)
    db_manager.create_all()
    return db_manager

def test_user_creation(db):
    def test_logic():
        user = User(name="Test", email="test@example.com")
        user.create()

        retrieved = User.get(user.id)
        assert retrieved.name == "Test"

    db.with_db_session(test_logic)()
```

## Troubleshooting

### Session Errors

```
AttributeError: db_session not set in the current context
```

**Solution**: Always use `db_manager.with_db_session()` wrapper.

### Type Mapping Issues

GraphQL DB automatically maps SQLAlchemy types to GraphQL. See [Models & Schema](models-schema/) for supported types.

### Import Errors

Ensure all dependencies are installed:

```bash
pip install graphql-db graphql-api sqlalchemy sqlalchemy-utils
```
