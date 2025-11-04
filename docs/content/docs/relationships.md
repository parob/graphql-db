---
title: "Relationships"
weight: 4
---

# Relationships

GraphQL DB seamlessly handles SQLAlchemy relationships in GraphQL queries.

## One-to-Many

Define relationships in your models:

```python
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str]
    posts = relationship("Post", back_populates="author")

class Post(ModelBase):
    __tablename__ = 'posts'

    title: Mapped[str]
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id'))
    author = relationship("User", back_populates="posts")
```

Query relationships:

```graphql
query {
  users {
    name
    posts {
      title
    }
  }
}
```

## Many-to-Many

Use association tables:

```python
from sqlalchemy import Table, Column

user_roles = Table(
    'user_roles',
    ModelBase.metadata,
    Column('user_id', UUID, ForeignKey('users.id')),
    Column('role_id', UUID, ForeignKey('roles.id'))
)

class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str]
    roles = relationship("Role", secondary=user_roles, back_populates="users")

class Role(ModelBase):
    __tablename__ = 'roles'

    name: Mapped[str]
    users = relationship("User", secondary=user_roles, back_populates="roles")
```

## Eager Loading

Prevent N+1 queries with eager loading:

```python
from sqlalchemy.orm import selectinload

@api.field
def users_with_posts(self) -> list[User]:
    """Get users with posts eagerly loaded."""
    return User.query().options(selectinload(User.posts)).all()

@api.field
def posts_with_authors(self) -> list[Post]:
    """Get posts with authors eagerly loaded."""
    return Post.query().options(selectinload(Post.author)).all()
```

Learn more in the [Performance](performance/) guide.
