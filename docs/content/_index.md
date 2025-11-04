---
title: "GraphQL DB for Python"
type: docs
---

> **SQLAlchemy integration for GraphQL APIs with automatic schema generation, query optimization, and database features.**

# GraphQL DB for Python

[![PyPI version](https://badge.fury.io/py/graphql-db.svg)](https://badge.fury.io/py/graphql-db)
[![Python versions](https://img.shields.io/pypi/pyversions/graphql-db.svg)](https://pypi.org/project/graphql-db/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## What is GraphQL DB?

GraphQL DB extends [graphql-api](https://graphql-api.parob.com/) with powerful SQLAlchemy integration. It provides automatic GraphQL schema generation from database models, query optimization, pagination, filtering, and relationship handling—making it effortless to build database-backed GraphQL APIs while maintaining high performance.

## Why GraphQL DB?

| Feature | Description |
|---------|-------------|
| 🗄️ **SQLAlchemy 2.0+ Integration** | Full support for modern SQLAlchemy with `Mapped[]` type annotations. |
| 🚀 **Automatic Schema Generation** | Database models become GraphQL types automatically. |
| 📄 **Relay Pagination** | Built-in cursor-based pagination following Relay specifications. |
| 🔍 **Query Optimization** | Automatic N+1 query prevention and relationship loading. |
| 🎯 **Advanced Filtering** | Powerful filtering system for complex database queries. |
| 📊 **Performance Optimized** | Efficient query patterns for large datasets. |

## Quick Start

Install GraphQL DB with dependencies:

```bash
pip install graphql-db graphql-api sqlalchemy
```

Define your database models:

```python
from graphql_api import GraphQLAPI
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from graphql_db.orm_base import DatabaseManager, ModelBase

# Initialize database
db_manager = DatabaseManager(url="sqlite:///example.db")

# Define your model
class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)

# Create tables
db_manager.create_all()

# Create GraphQL API
api = GraphQLAPI()

@api.type(is_root_type=True)
class Query:
    @api.field
    def users(self) -> list[User]:
        """Get all users."""
        return User.query().all()

# Execute with session management
def run_query():
    return api.execute('{ users { name email } }')

result = db_manager.with_db_session(run_query)()
```

> **Learn more**: See [graphql-api documentation](https://graphql-api.parob.com/) for GraphQL basics.

## How It Works

GraphQL DB automatically:

1. **Maps Models** - SQLAlchemy models become GraphQL types
2. **Optimizes Queries** - Prevents N+1 queries with smart loading
3. **Manages Sessions** - Handles database sessions via context managers
4. **Enables Pagination** - Provides Relay-style cursor pagination
5. **Supports Relationships** - Seamlessly handles one-to-many and many-to-many

## Key Features

### Automatic Type Mapping

SQLAlchemy models are automatically converted to GraphQL types:

```python
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

class Post(ModelBase):
    __tablename__ = 'posts'

    title: Mapped[str] = mapped_column(String(200))
    published: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(DateTime)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id'))

    author = relationship("User", back_populates="posts")
```

This model automatically becomes a GraphQL type with appropriate field types.

### Relay Pagination

Built-in support for Relay-style cursor pagination:

```python
from graphql_db.relay_base import relay_connection

UserConnection = relay_connection(User)

@api.field
def users_connection(
    self,
    first: int | None = 10,
    after: str | None = None
) -> UserConnection:
    return UserConnection(model=User, first=first, after=after)
```

### Session Management

Automatic database session handling:

```python
# Sessions managed automatically
def execute_query():
    result = api.execute('{ users { name } }')
    return result

# Wrap with session context
result = db_manager.with_db_session(execute_query)()
```

## Related Projects

GraphQL DB integrates with the GraphQL ecosystem:

### Core: graphql-api

Build GraphQL schemas with [graphql-api](https://graphql-api.parob.com/):

```python
from graphql_api import GraphQLAPI

# GraphQL DB extends graphql-api with database features
api = GraphQLAPI()
```

### Serving: graphql-http

Serve your database API over HTTP with [graphql-http](https://graphql-http.parob.com/):

```python
from graphql_http import GraphQLHTTP

server = GraphQLHTTP.from_api(api)
server.run()
```

### MCP Tools: graphql-mcp

Expose as MCP tools with [graphql-mcp](https://graphql-mcp.parob.com/):

```python
from graphql_mcp.server import GraphQLMCP

server = GraphQLMCP.from_api(api)
app = server.http_app()
```

## What's Next?

- 📚 **[Getting Started](docs/getting-started/)** - Learn the basics
- 🔧 **[Models & Schema](docs/models-schema/)** - Define database models
- 📄 **[Pagination](docs/pagination/)** - Implement Relay pagination
- ⚡ **[Performance](docs/performance/)** - Optimize queries
- 💡 **[Examples](docs/examples/)** - Real-world examples
- 📖 **[API Reference](docs/api-reference/)** - Complete API documentation
