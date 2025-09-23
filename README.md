# GraphQL Database Mapper

Generate GraphQL APIs from database models using SQLAlchemy 2.0.

## Features

- 🚀 Modern SQLAlchemy 2.0 support with `Mapped[]` annotations
- 🔗 Automatic GraphQL schema generation from database models
- 📊 Built-in database session management
- 🔍 UUID primary key support
- 🧪 Comprehensive test coverage

## Installation

```bash
uv add graphql-db
```

## Quick Start

```python
from graphql_db import DatabaseManager, ModelBase
from graphql_api import GraphQLAPI
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import String

# Define your model
class User(ModelBase):
    __tablename__ = 'users'
    name: Mapped[str] = mapped_column(String(50))

# Create GraphQL API
schema = GraphQLAPI()

@schema.type(is_root_type=True)
class Query:
    @schema.field
    def users(self) -> list[User]:
        return User.query().all()

# Setup database
db_manager = DatabaseManager()
```
