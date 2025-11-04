---
title: "Models & Schema"
weight: 2
---

# Models & Schema

GraphQL DB automatically converts SQLAlchemy models into GraphQL types using [graphql-api](https://graphql-api.parob.com/).

## Defining Models

Use SQLAlchemy 2.0+ with `Mapped[]` annotations:

```python
from datetime import datetime
from sqlalchemy import String, DateTime, Boolean
from sqlalchemy.orm import Mapped, mapped_column
from graphql_db.orm_base import ModelBase

class Post(ModelBase):
    __tablename__ = 'posts'

    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(String(1000))
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)
```

This model automatically becomes a GraphQL type.

## Type Mapping

| SQLAlchemy Type | Python Type | GraphQL Type |
|-----------------|-------------|--------------|
| `String` | `str` | `String` |
| `Integer` | `int` | `Int` |
| `Float` | `float` | `Float` |
| `Boolean` | `bool` | `Boolean` |
| `UUID` | `uuid.UUID` | `ID` |
| `DateTime` | `datetime` | `DateTime` |
| `JSON` | `dict` | `JSON` |

## Optional Fields

Use `Optional[]` or `| None` for nullable fields:

```python
from typing import Optional

class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str]
    bio: Mapped[str | None]  # Optional field
    age: Mapped[Optional[int]]  # Also optional
```

## Model Methods

`ModelBase` provides convenient methods:

```python
# Query
users = User.query().all()
active_users = User.query().filter(User.active == True).all()

# Get by ID
user = User.get(user_id)

# Create/Update
user = User(name="Alice", email="alice@example.com")
user.create()

# Delete
user.delete()
```

## Using Models in GraphQL

Simply return models from your GraphQL fields:

```python
from graphql_api import GraphQLAPI

api = GraphQLAPI()

@api.type(is_root_type=True)
class Query:
    @api.field
    def users(self) -> list[User]:
        return User.query().all()

    @api.field
    def posts(self, published_only: bool = False) -> list[Post]:
        query = Post.query()
        if published_only:
            query = query.filter(Post.published == True)
        return query.all()
```

For more on GraphQL schema definition, see the [graphql-api documentation](https://graphql-api.parob.com/docs/fundamentals/defining-schemas/).
