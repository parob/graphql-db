---
title: "Performance"
weight: 5
---

# Performance Optimization

GraphQL DB provides tools for building high-performance database-backed GraphQL APIs.

## Preventing N+1 Queries

Use eager loading to avoid N+1 query problems:

```python
from sqlalchemy.orm import selectinload, joinedload

@api.field
def users(self) -> list[User]:
    """Eager load posts to prevent N+1 queries."""
    return User.query().options(
        selectinload(User.posts)
    ).all()
```

### selectinload vs joinedload

- **`selectinload`**: Separate SELECT for relationships (better for collections)
- **`joinedload`**: Single SELECT with JOIN (better for single relationships)

```python
# Use selectinload for collections
User.query().options(selectinload(User.posts)).all()

# Use joinedload for single relationships
Post.query().options(joinedload(Post.author)).all()
```

## Connection Pooling

Configure connection pooling for production:

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,
    pool_recycle=3600
)

db_manager.engine = engine
```

## Query Optimization

### Use Indexes

```python
from sqlalchemy import Index

class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str]
    email: Mapped[str]

    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_name', 'name'),
    )
```

### Limit Results

Always limit query results:

```python
@api.field
def posts(self, limit: int = 100) -> list[Post]:
    """Limit results to prevent loading entire table."""
    if limit > 1000:  # Cap maximum limit
        limit = 1000
    return Post.query().limit(limit).all()
```

### Use Pagination

For large datasets, always use pagination. See the [Pagination](pagination/) guide.

## Batch Operations

For bulk operations, use SQLAlchemy's bulk methods:

```python
# Bulk insert
users = [
    User(name=f"User {i}", email=f"user{i}@example.com")
    for i in range(1000)
]

from sqlalchemy.orm import Session
session = Session(db_manager.engine)
session.bulk_save_objects(users)
session.commit()
```

## Monitoring

Log slow queries in development:

```python
import logging

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
```

## Best Practices

1. **Always eager load relationships** accessed in GraphQL queries
2. **Use connection pooling** in production
3. **Add indexes** to frequently queried columns
4. **Implement pagination** for large result sets
5. **Limit query results** to prevent memory issues
6. **Monitor query performance** and optimize slow queries
7. **Use batch operations** for bulk inserts/updates

See also the [graphql-api performance guide](https://graphql-api.parob.com/docs/advanced-features/performance/) for general GraphQL optimization tips.
