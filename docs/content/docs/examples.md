---
title: "Examples"
weight: 6
---

# Examples

Complete examples of using GraphQL DB.

## Blog API

A complete blog API with posts, users, and comments:

```python
from datetime import datetime
from graphql_api import GraphQLAPI
from graphql_db.orm_base import DatabaseManager, ModelBase
from sqlalchemy import String, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

# Database setup
db_manager = DatabaseManager(url="sqlite:///blog.db")

# Models
class User(ModelBase):
    __tablename__ = 'users'

    name: Mapped[str] = mapped_column(String(100))
    email: Mapped[str] = mapped_column(String(100), unique=True)
    posts = relationship("Post", back_populates="author")

class Post(ModelBase):
    __tablename__ = 'posts'

    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('users.id'))

    author = relationship("User", back_populates="posts")

db_manager.create_all()

# GraphQL API
api = GraphQLAPI()

@api.type(is_root_type=True)
class Root:
    @api.field
    def users(self) -> list[User]:
        from sqlalchemy.orm import selectinload
        return User.query().options(selectinload(User.posts)).all()

    @api.field
    def posts(self, published_only: bool = False) -> list[Post]:
        from sqlalchemy.orm import selectinload
        query = Post.query().options(selectinload(Post.author))
        if published_only:
            query = query.filter(Post.published == True)
        return query.all()

    @api.field(mutable=True)
    def create_user(self, name: str, email: str) -> User:
        user = User(name=name, email=email)
        user.create()
        return user

    @api.field(mutable=True)
    def create_post(self, title: str, content: str, author_id: str) -> Post:
        import uuid
        post = Post(
            title=title,
            content=content,
            author_id=uuid.UUID(author_id)
        )
        post.create()
        return post
```

## With HTTP Server

Serve the blog API over HTTP using [graphql-http](https://graphql-http.parob.com/):

```python
from graphql_http import GraphQLHTTP

server = GraphQLHTTP.from_api(api)

# Add database session middleware
@server.app.middleware("http")
async def db_session_middleware(request, call_next):
    def handler():
        return call_next(request)
    return await db_manager.with_db_session(handler)()

if __name__ == "__main__":
    server.run(host="0.0.0.0", port=8000)
```

## With Pagination

Add Relay-style pagination:

```python
from graphql_db.relay_base import relay_connection

PostConnection = relay_connection(Post)

@api.field
def posts_connection(
    self,
    first: int | None = 10,
    after: str | None = None
) -> PostConnection:
    from sqlalchemy.orm import selectinload
    query = Post.query().options(selectinload(Post.author))
    return PostConnection(
        model=Post,
        first=first,
        after=after,
        base_query=query
    )
```

## Testing

Test your database API:

```python
import pytest
from graphql_db.orm_base import DatabaseManager

@pytest.fixture
def db():
    db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)
    db_manager.create_all()
    return db_manager

def test_create_user(db):
    def test_logic():
        # Create user via mutation
        result = api.execute('''
            mutation {
                createUser(name: "Test User", email: "test@example.com") {
                    id
                    name
                    email
                }
            }
        ''')

        assert not result.errors
        assert result.data['createUser']['name'] == "Test User"

        # Query user
        result = api.execute('''
            query {
                users {
                    name
                    email
                }
            }
        ''')

        assert len(result.data['users']) == 1
        assert result.data['users'][0]['name'] == "Test User"

    db.with_db_session(test_logic)()
```

## More Examples

See the test suite in the repository for comprehensive examples:

- **CRUD Operations**: Basic create, read, update, delete
- **Relationships**: One-to-many and many-to-many
- **Pagination**: Relay-style cursors
- **Performance**: Query optimization

For general GraphQL patterns, see the [graphql-api examples](https://graphql-api.parob.com/docs/reference/examples/).
