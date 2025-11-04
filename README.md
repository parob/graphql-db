# GraphQL-DB

[![PyPI version](https://badge.fury.io/py/graphql-db.svg)](https://badge.fury.io/py/graphql-db)
[![Python versions](https://img.shields.io/pypi/pyversions/graphql-db.svg)](https://pypi.org/project/graphql-db/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[📚 Documentation](https://graphql-db.parob.com/)** | **[📦 PyPI](https://pypi.org/project/graphql-db/)** | **[🔧 GitHub](https://github.com/parob/graphql-db)**

---

SQLAlchemy integration for [graphql-api](https://graphql-api.parob.com/) with automatic schema generation, query optimization, and database features.

## Features

- 🗄️ **SQLAlchemy 2.0+ Integration** - Full support for modern SQLAlchemy
- 🚀 **Automatic Schema Generation** - Database models become GraphQL types
- 📄 **Relay Pagination** - Built-in cursor-based pagination
- 🔍 **Query Optimization** - Automatic N+1 prevention
- 🎯 **Advanced Filtering** - Powerful filtering for complex queries
- 📊 **Performance Optimized** - Efficient patterns for large datasets

## Installation

```bash
pip install graphql-db graphql-api sqlalchemy
```

## Quick Start

```python
from graphql_api import GraphQLAPI
from graphql_db.orm_base import DatabaseManager, ModelBase
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

# Initialize database
db_manager = DatabaseManager(url="sqlite:///myapp.db")

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

## Documentation

**Visit the [official documentation](https://graphql-db.parob.com/)** for comprehensive guides, examples, and API reference.

### Key Topics

- **[Getting Started](https://graphql-db.parob.com/docs/getting-started/)** - Quick introduction and basic usage
- **[Models & Schema](https://graphql-db.parob.com/docs/models-schema/)** - Define database models
- **[Pagination](https://graphql-db.parob.com/docs/pagination/)** - Implement Relay pagination
- **[Relationships](https://graphql-db.parob.com/docs/relationships/)** - Handle model relationships
- **[Performance](https://graphql-db.parob.com/docs/performance/)** - Query optimization
- **[Examples](https://graphql-db.parob.com/docs/examples/)** - Real-world usage examples
- **[API Reference](https://graphql-db.parob.com/docs/api-reference/)** - Complete API documentation

## Related Projects

GraphQL DB integrates with the GraphQL ecosystem:

- **[graphql-api](https://graphql-api.parob.com/)** - Core GraphQL schema building (required)
- **[graphql-http](https://graphql-http.parob.com/)** - Serve your database API over HTTP
- **[graphql-mcp](https://graphql-mcp.parob.com/)** - Expose as MCP tools for AI agents

## Key Features

### Automatic Type Mapping

SQLAlchemy models automatically become GraphQL types with proper field type mapping.

### Relay Pagination

Built-in support for Relay-style cursor pagination for efficient handling of large datasets.

### Session Management

Automatic database session handling via context managers prevents session management errors.

### Query Optimization

Eager loading support prevents N+1 queries and optimizes relationship loading.

See the [documentation](https://graphql-db.parob.com/) for detailed guides and examples.

## License

MIT License - see [LICENSE](LICENSE) file for details.
