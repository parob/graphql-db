# GraphQL-DB

SQLAlchemy integration for GraphQL-API with automatic schema generation, query optimization, and advanced database features.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

GraphQL-DB extends GraphQL-API with powerful SQLAlchemy integration, providing automatic GraphQL schema generation from database models, query optimization, pagination, filtering, and relationship handling. It's designed to make building database-backed GraphQL APIs effortless while maintaining high performance.

## Key Features

- **🗄️ SQLAlchemy 2.0+ Integration**: Full support for modern SQLAlchemy with `Mapped[]` type annotations
- **🚀 Automatic Schema Generation**: Database models become GraphQL types automatically
- **📄 Relay Pagination**: Built-in cursor-based pagination following Relay specifications
- **🔍 Smart Query Optimization**: Automatic N+1 query prevention and relationship loading
- **🎯 Advanced Filtering**: Powerful filtering system for complex database queries
- **📊 Performance Optimized**: Efficient query patterns for large datasets
- **🔄 Relationship Handling**: Seamless one-to-many and many-to-many relationships
- **🧪 Testing Support**: Comprehensive testing utilities with in-memory databases
- **🔧 Session Management**: Automatic database session handling with context managers

## Installation

```bash
pip install graphql-api sqlalchemy sqlalchemy-utils
```

**Additional dependencies for full functionality:**
```bash
pip install sqlmodel  # For SQLModel integration (optional)
```

## Quick Start

### Basic Database Model

```python
import uuid
from datetime import datetime
from typing import Optional, List
from graphql_api import GraphQLAPI
from sqlalchemy import String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Database setup (replace with your actual database)
from graphql_db.orm_base import DatabaseManager, ModelBase

# Initialize database
db_manager = DatabaseManager(url="sqlite:///example.db")

# Define your models
class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    posts = relationship("Post", back_populates="author")

class Post(ModelBase):
    __tablename__ = 'posts'

    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(String(1000))
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id'))

    # Relationships
    author = relationship("User", back_populates="posts")

# Create tables
db_manager.create_all()

# Create GraphQL API
api = GraphQLAPI()

@api.type(is_root_type=True)
class Query:
    @api.field
    def users(self) -> List[User]:
        """Get all users with their posts."""
        return User.query().all()

    @api.field
    def user(self, user_id: str) -> Optional[User]:
        """Get a specific user by ID."""
        return User.get(uuid.UUID(user_id))

    @api.field
    def posts(self, published_only: bool = False) -> List[Post]:
        """Get posts, optionally filtering by published status."""
        query = Post.query()
        if published_only:
            query = query.filter(Post.published.is_(True))
        return query.all()

# Execute queries with automatic session management
def run_query():
    result = api.execute('''
        query {
            users {
                name
                email
                posts {
                    title
                    published
                }
            }
        }
    ''')
    return result

# Use database session context
result = db_manager.with_db_session(run_query)()
```

### Mutations and CRUD Operations

```python
@api.type
class Mutation:
    @api.field
    def create_user(self, name: str, email: str) -> User:
        """Create a new user."""
        user = User(name=name, email=email)
        user.create()  # Automatically handles session
        return user

    @api.field
    def create_post(self, title: str, content: str, author_id: str) -> Post:
        """Create a new post."""
        post = Post(
            title=title,
            content=content,
            author_id=uuid.UUID(author_id)
        )
        post.create()
        return post

    @api.field
    def publish_post(self, post_id: str) -> Optional[Post]:
        """Publish a post."""
        post = Post.get(uuid.UUID(post_id))
        if post:
            post.published = True
            post.create()  # Updates existing record
        return post

    @api.field
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = User.get(uuid.UUID(user_id))
        if user:
            user.delete()
            return True
        return False

# Execute mutations
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

## Advanced Features

### Relay-Style Pagination

```python
from graphql_db.relay_base import relay_connection

# Create connection types
UserConnection = relay_connection(User)
PostConnection = relay_connection(Post)

@api.type(is_root_type=True)
class Query:
    @api.field
    def users_connection(
        self,
        first: Optional[int] = 10,
        after: Optional[str] = None,
        last: Optional[int] = None,
        before: Optional[str] = None
    ) -> UserConnection:
        """Get users with Relay-style pagination."""
        return UserConnection(
            model=User,
            first=first,
            after=after,
            last=last,
            before=before
        )

    @api.field
    def posts_connection(
        self,
        first: Optional[int] = 10,
        after: Optional[str] = None
    ) -> PostConnection:
        """Get posts with pagination."""
        return PostConnection(
            model=Post,
            first=first,
            after=after
        )

# Query with pagination
result = api.execute('''
    query {
        usersConnection(first: 5) {
            edges {
                node {
                    name
                    email
                }
                cursor
            }
            pageInfo {
                hasNextPage
                hasPreviousPage
                startCursor
                endCursor
            }
        }
    }
''')
```

### Advanced Filtering and Ordering

```python
from graphql_db.filter import Filter
from graphql_db.order_by import OrderBy, OrderByDirection

@api.type(is_root_type=True)
class Query:
    @api.field
    def filtered_posts(
        self,
        title_contains: Optional[str] = None,
        author_name: Optional[str] = None,
        published: Optional[bool] = None,
        order_by: str = "created_at",
        limit: int = 10
    ) -> List[Post]:
        """Get posts with advanced filtering and ordering."""
        query = Post.query()

        # Apply filters
        if title_contains:
            query = query.filter(Post.title.contains(title_contains))

        if published is not None:
            query = query.filter(Post.published.is_(published))

        if author_name:
            query = query.join(User).filter(User.name.contains(author_name))

        # Apply ordering
        if order_by == "title":
            query = query.order_by(Post.title.asc())
        elif order_by == "created_at":
            query = query.order_by(Post.created_at.desc())

        return query.limit(limit).all()

    @api.field
    def search_users(self, term: str) -> List[User]:
        """Search users by name or email."""
        return User.query().filter(
            (User.name.contains(term)) | (User.email.contains(term))
        ).all()
```

### Complex Relationships

```python
# Many-to-many relationship example
from sqlalchemy import Table

# Association table for many-to-many
user_roles = Table(
    'user_roles',
    ModelBase.metadata,
    Column('user_id', UUID, ForeignKey('users.id')),
    Column('role_id', UUID, ForeignKey('roles.id'))
)

class Role(ModelBase):
    __tablename__ = 'roles'

    name: Mapped[str] = mapped_column(String(50), unique=True)
    description: Mapped[Optional[str]] = mapped_column(String(200))

    # Many-to-many relationship
    users = relationship("User", secondary=user_roles, back_populates="roles")

class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)

    # Relationships
    posts = relationship("Post", back_populates="author")
    roles = relationship("Role", secondary=user_roles, back_populates="users")

@api.type(is_root_type=True)
class Query:
    @api.field
    def users_with_roles(self) -> List[User]:
        """Get users with their roles loaded."""
        from sqlalchemy.orm import selectinload
        return User.query().options(selectinload(User.roles)).all()
```

### Performance Optimization

```python
from sqlalchemy.orm import selectinload, joinedload

@api.type(is_root_type=True)
class Query:
    @api.field
    def optimized_posts(self) -> List[Post]:
        """Get posts with optimized loading to prevent N+1 queries."""
        return Post.query().options(
            selectinload(Post.author),  # Eager load authors
            selectinload(Post.comments)  # Eager load comments
        ).all()

    @api.field
    def posts_with_author_count(self) -> List[dict]:
        """Get posts with author post counts (using raw SQL)."""
        from sqlalchemy import func

        results = Post.query().join(User).add_columns(
            func.count(Post.id).label('author_post_count')
        ).group_by(User.id).all()

        return [
            {
                'post': post,
                'author_post_count': count
            }
            for post, count in results
        ]

    @api.field
    def paginated_posts(self, page: int = 1, per_page: int = 10) -> dict:
        """Efficient pagination for large datasets."""
        offset = (page - 1) * per_page

        posts = Post.query().offset(offset).limit(per_page).all()
        total_count = Post.query().count()

        return {
            'posts': posts,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': total_count,
                'pages': (total_count + per_page - 1) // per_page
            }
        }
```

### Custom Database Types

```python
from sqlalchemy import TypeDecorator, String
import json

class JSONType(TypeDecorator):
    """Custom JSON type for storing structured data."""
    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)

class UserProfile(ModelBase):
    __tablename__ = 'user_profiles'

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id'))
    preferences: Mapped[dict] = mapped_column(JSONType)
    settings: Mapped[dict] = mapped_column(JSONType)

    # Relationship
    user = relationship("User", backref="profile")

@api.type(is_root_type=True)
class Query:
    @api.field
    def user_with_profile(self, user_id: str) -> Optional[User]:
        """Get user with profile data."""
        return User.query().options(
            selectinload(User.profile)
        ).filter(User.id == uuid.UUID(user_id)).first()
```

## Testing Your Database API

### In-Memory Testing

```python
import pytest
from graphql_db.orm_base import DatabaseManager

@pytest.fixture
def db():
    """Create in-memory database for testing."""
    db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

    # Create tables
    db_manager.create_all()

    return db_manager

def test_user_creation(db):
    """Test user creation and retrieval."""
    def test_logic():
        # Create user
        user = User(name="Test User", email="test@example.com")
        user.create()

        # Retrieve user
        retrieved = User.get(user.id)
        assert retrieved is not None
        assert retrieved.name == "Test User"
        assert retrieved.email == "test@example.com"

        # Test query
        all_users = User.query().all()
        assert len(all_users) == 1

    db.with_db_session(test_logic)()

def test_post_with_author(db):
    """Test post creation with author relationship."""
    def test_logic():
        # Create user
        user = User(name="Author", email="author@example.com")
        user.create()

        # Create post
        post = Post(
            title="Test Post",
            content="Test content",
            author_id=user.id
        )
        post.create()

        # Test relationship
        retrieved_post = Post.get(post.id)
        assert retrieved_post.author.name == "Author"

        # Test reverse relationship
        assert len(user.posts) == 1
        assert user.posts[0].title == "Test Post"

    db.with_db_session(test_logic)()

def test_graphql_integration(db):
    """Test GraphQL queries with database."""
    api = GraphQLAPI()

    @api.type(is_root_type=True)
    class Query:
        @api.field
        def users(self) -> List[User]:
            return User.query().all()

    def test_logic():
        # Create test data
        user = User(name="GraphQL User", email="graphql@example.com")
        user.create()

        # Execute GraphQL query
        result = api.execute('''
            query {
                users {
                    name
                    email
                }
            }
        ''')

        assert not result.errors
        assert len(result.data['users']) == 1
        assert result.data['users'][0]['name'] == "GraphQL User"

    db.with_db_session(test_logic)()
```

### Performance Testing

```python
def test_bulk_operations(db):
    """Test performance with large datasets."""
    def test_logic():
        import time

        # Bulk create users
        start_time = time.time()
        for i in range(1000):
            user = User(name=f"User {i}", email=f"user{i}@example.com")
            user.create()
        creation_time = time.time() - start_time
        print(f"Created 1000 users in {creation_time:.2f} seconds")

        # Test query performance
        start_time = time.time()
        all_users = User.query().all()
        query_time = time.time() - start_time
        print(f"Queried {len(all_users)} users in {query_time:.3f} seconds")

        # Test filtered query
        start_time = time.time()
        filtered_users = User.query().filter(
            User.name.contains("User 1")
        ).all()
        filter_time = time.time() - start_time
        print(f"Filtered query returned {len(filtered_users)} users in {filter_time:.3f} seconds")

    db.with_db_session(test_logic)()
```

## SQLModel Integration

For projects using SQLModel, GraphQL-DB provides seamless integration:

```python
from sqlmodel import SQLModel, Field, Session, select
from typing import Optional, List

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    email: str

class Post(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    content: str
    author_id: Optional[int] = Field(default=None, foreign_key="user.id")

# GraphQL integration
@api.type(is_root_type=True)
class Query:
    @api.field
    def users(self) -> List[User]:
        """Get all users using SQLModel."""
        statement = select(User)
        return session.exec(statement).all()

    @api.field
    def posts_with_authors(self) -> List[Post]:
        """Get posts with eager loading."""
        from sqlalchemy.orm import selectinload
        statement = select(Post).options(selectinload(Post.author))
        return session.exec(statement).all()
```

## Database Configuration

### Production Configuration

```python
from graphql_db.orm_base import DatabaseManager
import os

# Environment-based configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://user:password@localhost/mydb'
)

db_manager = DatabaseManager(
    url=DATABASE_URL,
    install=True,  # Create tables if they don't exist
    wipe=False     # Don't drop existing tables
)

# Connection pooling for production
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True,  # Validate connections
    pool_recycle=3600    # Recycle connections every hour
)

db_manager.engine = engine
```

### Migration Support

```python
# Database migrations with Alembic
def setup_migrations():
    """Set up Alembic for database migrations."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")

    # Generate migration
    command.revision(alembic_cfg, autogenerate=True, message="Auto migration")

    # Apply migrations
    command.upgrade(alembic_cfg, "head")

# Version checking
def check_database_version():
    """Check if database schema is up to date."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.current(alembic_cfg, verbose=True)
```

## Best Practices

### 1. Use Proper Session Management

```python
# Good - Use context manager
def get_users():
    return db_manager.with_db_session(lambda: User.query().all())()

# Bad - Manual session handling
def get_users_bad():
    session = db_manager.session()
    try:
        return session.query(User).all()
    finally:
        session.close()
```

### 2. Optimize Relationships

```python
# Good - Eager loading
@api.field
def posts_with_authors(self) -> List[Post]:
    return Post.query().options(selectinload(Post.author)).all()

# Bad - Lazy loading (N+1 problem)
@api.field
def posts_with_authors_bad(self) -> List[Post]:
    posts = Post.query().all()
    # This will cause N+1 queries when accessing post.author
    return posts
```

### 3. Use Proper Error Handling

```python
from graphql_api import GraphQLError
from sqlalchemy.exc import IntegrityError

@api.field
def create_user(self, name: str, email: str) -> User:
    try:
        user = User(name=name, email=email)
        user.create()
        return user
    except IntegrityError as e:
        if "unique constraint" in str(e).lower():
            raise GraphQLError(f"User with email {email} already exists")
        raise GraphQLError("Database error occurred")
```

### 4. Implement Pagination for Large Datasets

```python
@api.field
def posts(self, page: int = 1, per_page: int = 10) -> dict:
    """Always paginate large datasets."""
    if per_page > 100:  # Limit page size
        per_page = 100

    offset = (page - 1) * per_page
    posts = Post.query().offset(offset).limit(per_page).all()
    total = Post.query().count()

    return {
        'items': posts,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': total,
            'has_next': offset + per_page < total
        }
    }
```

### 5. Use Database Indexes

```python
from sqlalchemy import Index

class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Add indexes for frequently queried fields
    __table_args__ = (
        Index('idx_user_email', 'email'),
        Index('idx_user_created_at', 'created_at'),
        Index('idx_user_name_email', 'name', 'email'),  # Composite index
    )
```

## API Reference

### Core Classes

- **`DatabaseManager`**: Main class for database connection and session management
- **`ModelBase`**: Base class for all database models with built-in GraphQL integration
- **`relay_connection()`**: Function to create Relay-style pagination connections

### ModelBase Methods

- **`query(session=None)`**: Get a query builder for the model
- **`filter(*args, **kwargs)`**: Filter query with conditions
- **`get(id, session=None)`**: Get a single instance by ID
- **`create(session=None)`**: Save instance to database
- **`delete(session=None)`**: Mark instance for deletion

### Database Types

| Python Type | SQLAlchemy Type | GraphQL Type |
|-------------|----------------|--------------|
| `str` | `String` | `String` |
| `int` | `Integer` | `Int` |
| `float` | `Float` | `Float` |
| `bool` | `Boolean` | `Boolean` |
| `uuid.UUID` | `UUID`/`UUIDType` | `ID` |
| `datetime` | `DateTime` | `DateTime` |
| `date` | `Date` | `Date` |
| `dict` | `JSON` | `JSON` |

## Troubleshooting

### Common Issues

#### 1. Session Management Errors
```python
# Problem: Session not in context
AttributeError: db_session not set in the current context

# Solution: Use session context
db_manager.with_db_session(your_function)()
```

#### 2. Relationship Loading Issues
```python
# Problem: Lazy loading after session closed
DetachedInstanceError: Instance is not bound to a Session

# Solution: Eager load relationships
Post.query().options(selectinload(Post.author)).all()
```

#### 3. UUID Handling
```python
# Problem: Invalid UUID format
ValueError: badly formed hexadecimal UUID string

# Solution: Validate UUIDs
try:
    user_id = uuid.UUID(user_id_string)
    user = User.get(user_id)
except ValueError:
    raise GraphQLError("Invalid user ID format")
```

## Performance Tips

1. **Use Connection Pooling**: Configure appropriate pool sizes for your workload
2. **Implement Caching**: Cache frequent queries using Redis or Memcached
3. **Optimize Queries**: Use `explain()` to analyze query performance
4. **Batch Operations**: Use bulk operations for large datasets
5. **Index Strategy**: Create indexes for frequently filtered/sorted columns
6. **Pagination**: Always paginate large result sets
7. **Monitor Queries**: Log slow queries and optimize them

## Examples Repository

Complete working examples are available in the test suite:

- **Basic CRUD Operations**: `tests/test_integration.py`
- **Pagination**: `tests/test_pagination.py`
- **Relationships**: `tests/test_integration.py`
- **Performance**: `tests/test_pagination.py`
- **Edge Cases**: `tests/test_edge_cases.py`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/`
5. Run linting: `flake8`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 **Email**: rob@parob.com
- 🐛 **Issues**: [GitLab Issues](https://gitlab.com/parob/graphql-api/issues)
- 📖 **Documentation**: See test files for comprehensive examples
- 💬 **Community**: Join our discussions for help and best practices

---

**Ready to build database-powered GraphQL APIs? Start with the Quick Start guide above! 🗄️🚀**