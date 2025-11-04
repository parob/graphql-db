---
title: "API Reference"
weight: 7
---

# API Reference

Complete API documentation for GraphQL DB.

## DatabaseManager

Main class for database connection and session management.

### Constructor

```python
DatabaseManager(
    url: str,
    install: bool = False,
    wipe: bool = False
)
```

**Parameters:**

- `url` (str): SQLAlchemy database URL
- `install` (bool): Create tables if they don't exist
- `wipe` (bool): Drop all tables before creating

**Example:**

```python
db_manager = DatabaseManager(
    url="postgresql://user:pass@localhost/mydb",
    install=True
)
```

### Methods

#### create_all()

Create all database tables from model definitions.

```python
db_manager.create_all()
```

#### with_db_session()

Context manager for automatic session handling.

```python
def my_function():
    return User.query().all()

result = db_manager.with_db_session(my_function)()
```

**Parameters:**

- `func` (Callable): Function to execute within session context

**Returns:** Wrapped function that manages database session

## ModelBase

Base class for all database models.

### Class Methods

#### query()

Get a query builder for the model.

```python
users = User.query().filter(User.active == True).all()
```

**Returns:** SQLAlchemy Query object

#### get()

Get a single instance by ID.

```python
user = User.get(user_id)
```

**Parameters:**

- `id` (UUID): Model instance ID

**Returns:** Model instance or None

### Instance Methods

#### create()

Save or update the instance in the database.

```python
user = User(name="Alice", email="alice@example.com")
user.create()
```

#### delete()

Mark instance for deletion from database.

```python
user.delete()
```

## relay_connection()

Create a Relay-style pagination connection for a model.

```python
from graphql_db.relay_base import relay_connection

UserConnection = relay_connection(User)
```

**Parameters:**

- `model` (Type[ModelBase]): Model class to create connection for

**Returns:** Connection class with edges, nodes, and pageInfo

### Connection Parameters

When using a connection in GraphQL:

- `first` (int, optional): Number of items to fetch
- `after` (str, optional): Cursor for forward pagination
- `last` (int, optional): Number of items from end
- `before` (str, optional): Cursor for backward pagination

**Example:**

```python
@api.field
def users_connection(
    self,
    first: int | None = 10,
    after: str | None = None
) -> UserConnection:
    return UserConnection(model=User, first=first, after=after)
```

## Type Mappings

| Python Type | SQLAlchemy Type | GraphQL Type |
|-------------|----------------|--------------|
| `str` | `String` | `String` |
| `int` | `Integer` | `Int` |
| `float` | `Float` | `Float` |
| `bool` | `Boolean` | `Boolean` |
| `uuid.UUID` | `UUID` | `ID` |
| `datetime` | `DateTime` | `DateTime` |
| `date` | `Date` | `Date` |
| `dict` | `JSON` | `JSON` |
| `list[T]` | - | `[T]` |
| `T | None` | - | `T` (nullable) |

## SQLAlchemy Integration

GraphQL DB works with SQLAlchemy 2.0+ features:

- `Mapped[]` type annotations
- `mapped_column()` for column definitions
- `relationship()` for model relationships
- Modern query API with `select()`

For more on SQLAlchemy, see the [SQLAlchemy documentation](https://docs.sqlalchemy.org/).

For GraphQL schema building, see the [graphql-api API reference](https://graphql-api.parob.com/docs/reference/api-reference/).
