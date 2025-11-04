---
title: "Pagination"
weight: 3
---

# Pagination

GraphQL DB provides Relay-style cursor-based pagination for efficient handling of large datasets.

## Relay Connections

Create paginated connections for your models:

```python
from graphql_db.relay_base import relay_connection

# Create connection type
UserConnection = relay_connection(User)

@api.field
def users_connection(
    self,
    first: int | None = 10,
    after: str | None = None,
    last: int | None = None,
    before: str | None = None
) -> UserConnection:
    """Get paginated users."""
    return UserConnection(
        model=User,
        first=first,
        after=after,
        last=last,
        before=before
    )
```

## Querying Paginated Data

```graphql
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
```

## Forward Pagination

Fetch the next page:

```python
# First page
first_result = api.execute('''
    query {
        usersConnection(first: 10) {
            edges { node { name } cursor }
            pageInfo { endCursor hasNextPage }
        }
    }
''')

# Get end cursor for next page
end_cursor = first_result.data['usersConnection']['pageInfo']['endCursor']

# Next page
next_result = api.execute(f'''
    query {{
        usersConnection(first: 10, after: "{end_cursor}") {{
            edges {{ node {{ name }} cursor }}
            pageInfo {{ endCursor hasNextPage }}
        }}
    }}
''')
```

## Backward Pagination

Fetch the previous page:

```python
usersConnection(last: 10, before: "{cursor}")
```

## Simple Pagination

For simpler use cases, use offset/limit pagination:

```python
@api.field
def posts(self, page: int = 1, per_page: int = 10) -> dict:
    """Get posts with simple pagination."""
    offset = (page - 1) * per_page
    posts = Post.query().offset(offset).limit(per_page).all()
    total = Post.query().count()

    return {
        'items': posts,
        'page': page,
        'per_page': per_page,
        'total': total,
        'has_next': offset + per_page < total
    }
```

Learn more about [Relay pagination specification](https://relay.dev/graphql/connections.htm).
