"""Performance tests for graphql-db with deeply nested models.

These tests focus on N+1 query problems and database performance
with patterns inspired by the outeract codebase:
- Organisation -> Application -> User -> UserIdentity -> PlatformConnection
- Event -> Edge relationships
- WebhookSubscription -> WebhookDelivery chains

The tests measure query counts and execution time to identify bottlenecks.
"""
import time
import uuid
from typing import Optional, List
from datetime import datetime
from unittest.mock import patch
from contextlib import contextmanager

from graphql_api import GraphQLAPI
from sqlalchemy import Integer, String, ForeignKey, DateTime, Boolean, Text, event
from sqlalchemy.orm import Mapped, mapped_column, relationship, selectinload, joinedload
from sqlalchemy.engine import Engine

from graphql_db.orm_base import DatabaseManager, ModelBase


class QueryCounter:
    """Context manager to count SQL queries executed."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.count = 0
        self.queries: List[str] = []

    def _before_cursor_execute(self, conn, cursor, statement, parameters, context, executemany):
        self.count += 1
        self.queries.append(statement)

    def __enter__(self):
        event.listen(self.engine, "before_cursor_execute", self._before_cursor_execute)
        return self

    def __exit__(self, *args):
        event.remove(self.engine, "before_cursor_execute", self._before_cursor_execute)


class TestN1QueryProblem:
    """Tests demonstrating and measuring N+1 query problems."""

    def test_n_plus_1_with_blog_model(self):
        """Test N+1 problem with User -> Posts -> Comments pattern."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class User(ModelBase):
            __tablename__ = 'perf_users'
            username: Mapped[str] = mapped_column(String(50))
            email: Mapped[str] = mapped_column(String(100))
            posts = relationship("Post", back_populates="author", lazy="select")

        class Post(ModelBase):
            __tablename__ = 'perf_posts'
            title: Mapped[str] = mapped_column(String(200))
            content: Mapped[str] = mapped_column(Text)
            author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_users.id'))
            author = relationship("User", back_populates="posts")
            comments = relationship("Comment", back_populates="post", lazy="select")

        class Comment(ModelBase):
            __tablename__ = 'perf_comments'
            content: Mapped[str] = mapped_column(Text)
            post_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_posts.id'))
            post = relationship("Post", back_populates="comments")

        db_manager.base.metadata.create_all(db_manager.engine)

        def setup_data():
            # Create 10 users, each with 5 posts, each post with 3 comments
            for u in range(10):
                user = User(username=f"user_{u}", email=f"user_{u}@example.com")
                user.create()
                for p in range(5):
                    post = Post(
                        title=f"Post {p} by User {u}",
                        content=f"Content of post {p}",
                        author_id=user.id
                    )
                    post.create()
                    for c in range(3):
                        comment = Comment(content=f"Comment {c}", post_id=post.id)
                        comment.create()

        def test_without_eager_loading():
            """Without eager loading - demonstrates N+1 problem."""
            with QueryCounter(db_manager.engine) as counter:
                users = User.query().all()
                for user in users:
                    for post in user.posts:
                        _ = list(post.comments)  # Access comments

            print(f"\nWithout eager loading:")
            print(f"  Query count: {counter.count}")
            print(f"  Expected for N+1: 1 (users) + 10 (posts) + 50 (comments) = 61")
            return counter.count

        def test_with_eager_loading():
            """With eager loading - should be 1-3 queries."""
            from sqlalchemy.orm import selectinload

            with QueryCounter(db_manager.engine) as counter:
                users = User.query().options(
                    selectinload(User.posts).selectinload(Post.comments)
                ).all()
                for user in users:
                    for post in user.posts:
                        _ = list(post.comments)

            print(f"\nWith eager loading (selectinload):")
            print(f"  Query count: {counter.count}")
            print(f"  Expected: 3 (users, posts, comments)")
            return counter.count

        db_manager.with_db_session(setup_data)()
        n1_count = db_manager.with_db_session(test_without_eager_loading)()
        eager_count = db_manager.with_db_session(test_with_eager_loading)()

        # N+1 should have many more queries
        assert n1_count > 50, f"Expected N+1 to have 50+ queries, got {n1_count}"
        # Eager loading should be efficient
        assert eager_count <= 5, f"Expected eager loading to have <=5 queries, got {eager_count}"

        print(f"\nPerformance improvement: {n1_count / eager_count:.1f}x fewer queries")

    def test_outeract_like_5_level_nesting(self):
        """Test the outeract pattern: Org -> App -> User -> Identity -> PlatformConnection."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Organisation(ModelBase):
            __tablename__ = 'perf_orgs'
            name: Mapped[str] = mapped_column(String(100))
            applications = relationship("Application", back_populates="organisation", lazy="select")

        class Application(ModelBase):
            __tablename__ = 'perf_apps'
            name: Mapped[str] = mapped_column(String(100))
            org_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_orgs.id'))
            organisation = relationship("Organisation", back_populates="applications")
            users = relationship("AppUser", back_populates="application", lazy="select")
            platform_connections = relationship("PlatformConnection", back_populates="application", lazy="select")

        class AppUser(ModelBase):
            __tablename__ = 'perf_app_users'
            name: Mapped[str] = mapped_column(String(100))
            email: Mapped[str] = mapped_column(String(100))
            app_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_apps.id'))
            application = relationship("Application", back_populates="users")
            identities = relationship("UserIdentity", back_populates="user", lazy="select")

        class PlatformConnection(ModelBase):
            __tablename__ = 'perf_platform_connections'
            platform_name: Mapped[str] = mapped_column(String(50))
            app_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_apps.id'))
            application = relationship("Application", back_populates="platform_connections")
            identities = relationship("UserIdentity", back_populates="platform_connection", lazy="select")

        class UserIdentity(ModelBase):
            __tablename__ = 'perf_user_identities'
            external_id: Mapped[str] = mapped_column(String(200))
            identity_type: Mapped[str] = mapped_column(String(50))
            user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_app_users.id'))
            platform_connection_id: Mapped[Optional[uuid.UUID]] = mapped_column(
                ForeignKey('perf_platform_connections.id'), nullable=True
            )
            user = relationship("AppUser", back_populates="identities")
            platform_connection = relationship("PlatformConnection", back_populates="identities")

        db_manager.base.metadata.create_all(db_manager.engine)

        def setup_data():
            # Create realistic data: 3 orgs, 2 apps each, 5 users per app, 2 identities per user
            for o in range(3):
                org = Organisation(name=f"Organisation {o}")
                org.create()
                for a in range(2):
                    app = Application(name=f"App {a}", org_id=org.id)
                    app.create()

                    # Create platform connections first
                    platforms = []
                    for platform in ["whatsapp", "email", "instagram"]:
                        pc = PlatformConnection(platform_name=platform, app_id=app.id)
                        pc.create()
                        platforms.append(pc)

                    for u in range(5):
                        user = AppUser(
                            name=f"User {u}",
                            email=f"user_{u}@app_{a}.org_{o}.com",
                            app_id=app.id
                        )
                        user.create()
                        for i, platform in enumerate(platforms[:2]):
                            identity = UserIdentity(
                                external_id=f"ext_{user.id}_{i}",
                                identity_type=platform.platform_name,
                                user_id=user.id,
                                platform_connection_id=platform.id
                            )
                            identity.create()

        def test_n1_5_levels():
            """Test N+1 with 5 levels of nesting."""
            with QueryCounter(db_manager.engine) as counter:
                orgs = Organisation.query().all()
                for org in orgs:
                    for app in org.applications:
                        for user in app.users:
                            for identity in user.identities:
                                _ = identity.platform_connection

            print(f"\n5-level nesting without eager loading:")
            print(f"  Query count: {counter.count}")
            return counter.count

        def test_eager_5_levels():
            """Test with full eager loading for 5 levels."""
            with QueryCounter(db_manager.engine) as counter:
                orgs = Organisation.query().options(
                    selectinload(Organisation.applications)
                    .selectinload(Application.users)
                    .selectinload(AppUser.identities)
                    .selectinload(UserIdentity.platform_connection)
                ).all()
                for org in orgs:
                    for app in org.applications:
                        for user in app.users:
                            for identity in user.identities:
                                _ = identity.platform_connection

            print(f"\n5-level nesting with eager loading:")
            print(f"  Query count: {counter.count}")
            return counter.count

        db_manager.with_db_session(setup_data)()
        n1_count = db_manager.with_db_session(test_n1_5_levels)()
        eager_count = db_manager.with_db_session(test_eager_5_levels)()

        print(f"\nPerformance improvement: {n1_count / max(eager_count, 1):.1f}x fewer queries")

        # N+1 should have many queries (1 + 3 + 6 + 30 + 60 = 100+)
        assert n1_count > 50, f"Expected N+1 to have many queries, got {n1_count}"
        # Eager loading should be much more efficient
        assert eager_count <= 10, f"Expected eager loading to have <=10 queries, got {eager_count}"


class TestQueryExecutionTime:
    """Tests for actual query execution time."""

    def test_execution_time_with_large_dataset(self):
        """Test query execution time with a larger dataset."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Item(ModelBase):
            __tablename__ = 'perf_items'
            name: Mapped[str] = mapped_column(String(100))
            category: Mapped[str] = mapped_column(String(50))
            price: Mapped[int] = mapped_column(Integer)
            in_stock: Mapped[bool] = mapped_column(Boolean, default=True)

        db_manager.base.metadata.create_all(db_manager.engine)

        def setup_large_dataset():
            # Create 1000 items
            categories = ["electronics", "clothing", "food", "books", "toys"]
            for i in range(1000):
                item = Item(
                    name=f"Item {i}",
                    category=categories[i % len(categories)],
                    price=i * 100,
                    in_stock=i % 2 == 0
                )
                item.create()

        def test_query_performance():
            # Simple query
            start = time.perf_counter()
            items = Item.query().all()
            simple_time = (time.perf_counter() - start) * 1000
            assert len(items) == 1000

            # Filtered query
            start = time.perf_counter()
            electronics = Item.query().filter(Item.category == "electronics").all()
            filter_time = (time.perf_counter() - start) * 1000
            assert len(electronics) == 200

            # Complex filter
            start = time.perf_counter()
            expensive_in_stock = Item.query().filter(
                Item.price > 50000,
                Item.in_stock.is_(True)
            ).all()
            complex_time = (time.perf_counter() - start) * 1000

            print(f"\nQuery execution times (1000 items):")
            print(f"  Simple query (all): {simple_time:.2f}ms")
            print(f"  Filtered query (category): {filter_time:.2f}ms")
            print(f"  Complex filter (price + stock): {complex_time:.2f}ms")

            return simple_time, filter_time, complex_time

        db_manager.with_db_session(setup_large_dataset)()
        simple_time, filter_time, complex_time = db_manager.with_db_session(test_query_performance)()

        # All queries should complete in reasonable time
        assert simple_time < 500, f"Simple query too slow: {simple_time:.2f}ms"
        assert filter_time < 500, f"Filtered query too slow: {filter_time:.2f}ms"
        assert complex_time < 500, f"Complex query too slow: {complex_time:.2f}ms"


class TestGraphQLIntegrationPerformance:
    """Tests for GraphQL query execution with database."""

    def test_graphql_nested_query_performance(self):
        """Test GraphQL query execution with nested relationships."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Author(ModelBase):
            __tablename__ = 'perf_gql_authors'
            name: Mapped[str] = mapped_column(String(100))
            books = relationship("Book", back_populates="author", lazy="select")

        class Book(ModelBase):
            __tablename__ = 'perf_gql_books'
            title: Mapped[str] = mapped_column(String(200))
            author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_gql_authors.id'))
            author = relationship("Author", back_populates="books")
            reviews = relationship("Review", back_populates="book", lazy="select")

        class Review(ModelBase):
            __tablename__ = 'perf_gql_reviews'
            content: Mapped[str] = mapped_column(Text)
            rating: Mapped[int] = mapped_column(Integer)
            book_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_gql_books.id'))
            book = relationship("Book", back_populates="reviews")

        db_manager.base.metadata.create_all(db_manager.engine)

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Query:
            @schema.field
            def authors(self) -> list[Author]:
                # Without eager loading - demonstrates N+1
                return Author.query().all()

            @schema.field
            def authors_optimized(self) -> list[Author]:
                # With eager loading - optimized
                return Author.query().options(
                    selectinload(Author.books).selectinload(Book.reviews)
                ).all()

        def setup_data():
            for a in range(10):
                author = Author(name=f"Author {a}")
                author.create()
                for b in range(5):
                    book = Book(title=f"Book {b} by Author {a}", author_id=author.id)
                    book.create()
                    for r in range(3):
                        review = Review(
                            content=f"Review {r} for Book {b}",
                            rating=(r + a + b) % 5 + 1,
                            book_id=book.id
                        )
                        review.create()

        def test_graphql_queries():
            query = """
                query {
                    authors {
                        name
                        books {
                            title
                            reviews {
                                content
                                rating
                            }
                        }
                    }
                }
            """

            query_optimized = """
                query {
                    authorsOptimized {
                        name
                        books {
                            title
                            reviews {
                                content
                                rating
                            }
                        }
                    }
                }
            """

            # Test unoptimized
            start = time.perf_counter()
            with QueryCounter(db_manager.engine) as counter:
                result = schema.executor().execute(query)
            unoptimized_time = (time.perf_counter() - start) * 1000
            unoptimized_queries = counter.count

            assert result.errors is None

            # Test optimized
            start = time.perf_counter()
            with QueryCounter(db_manager.engine) as counter:
                result = schema.executor().execute(query_optimized)
            optimized_time = (time.perf_counter() - start) * 1000
            optimized_queries = counter.count

            assert result.errors is None

            print(f"\nGraphQL nested query performance:")
            print(f"  Unoptimized: {unoptimized_time:.2f}ms, {unoptimized_queries} queries")
            print(f"  Optimized: {optimized_time:.2f}ms, {optimized_queries} queries")
            print(f"  Query reduction: {unoptimized_queries / max(optimized_queries, 1):.1f}x")

            return unoptimized_queries, optimized_queries

        db_manager.with_db_session(setup_data)()
        unopt, opt = db_manager.with_db_session(test_graphql_queries)()

        # Optimized should have far fewer queries
        assert opt < unopt / 5, f"Optimization not effective: {opt} vs {unopt}"


class TestEventEdgePattern:
    """Tests for the Event/Edge pattern common in outeract."""

    def test_event_edge_query_performance(self):
        """Test querying events with edge relationships."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Event(ModelBase):
            __tablename__ = 'perf_events'
            event_type: Mapped[str] = mapped_column(String(100))
            payload: Mapped[str] = mapped_column(Text)  # JSON string
            processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
            edges = relationship("Edge", back_populates="source_event",
                               foreign_keys="Edge.source_event_id", lazy="select")

        class Edge(ModelBase):
            __tablename__ = 'perf_edges'
            source_event_id: Mapped[uuid.UUID] = mapped_column(ForeignKey('perf_events.id'))
            target_event_id: Mapped[Optional[uuid.UUID]] = mapped_column(
                ForeignKey('perf_events.id'), nullable=True
            )
            edge_type: Mapped[str] = mapped_column(String(50))
            source_event = relationship("Event", foreign_keys=[source_event_id], back_populates="edges")
            target_event = relationship("Event", foreign_keys=[target_event_id])

        db_manager.base.metadata.create_all(db_manager.engine)

        def setup_data():
            # Create conversation events with message events linked via edges
            conversations = []
            for c in range(10):
                conv = Event(
                    event_type="conversation.created",
                    payload=f'{{"conversation_id": "conv_{c}"}}'
                )
                conv.create()
                conversations.append(conv)

                # Create messages for each conversation
                for m in range(20):
                    msg = Event(
                        event_type="message.inbound",
                        payload=f'{{"message": "Message {m} in conv {c}"}}'
                    )
                    msg.create()

                    # Create edge linking message to conversation
                    edge = Edge(
                        source_event_id=msg.id,
                        target_event_id=conv.id,
                        edge_type="in_conversation"
                    )
                    edge.create()

        def test_event_queries():
            # Query all conversations with their message edges
            with QueryCounter(db_manager.engine) as counter:
                conversations = Event.query().filter(
                    Event.event_type == "conversation.created"
                ).all()

                # Access edges (triggers N+1)
                message_count = 0
                for conv in conversations:
                    edges = Edge.query().filter(
                        Edge.target_event_id == conv.id,
                        Edge.edge_type == "in_conversation"
                    ).all()
                    message_count += len(edges)

            print(f"\nEvent/Edge pattern (10 conversations, 20 messages each):")
            print(f"  Query count (N+1 pattern): {counter.count}")
            print(f"  Total messages found: {message_count}")

            return counter.count

        db_manager.with_db_session(setup_data)()
        query_count = db_manager.with_db_session(test_event_queries)()

        # With N+1, we expect at least 11 queries (1 for conversations + 10 for edges)
        assert query_count >= 11, f"Expected N+1 pattern, got {query_count} queries"


class TestRelayPaginationPerformance:
    """Tests for Relay-style pagination performance."""

    def test_pagination_with_large_dataset(self):
        """Test pagination performance with cursor-based pagination."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Message(ModelBase):
            __tablename__ = 'perf_messages'
            content: Mapped[str] = mapped_column(Text)
            created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

        db_manager.base.metadata.create_all(db_manager.engine)

        def setup_data():
            # Create 5000 messages
            for i in range(5000):
                msg = Message(content=f"Message {i}")
                msg.create()

        def test_pagination():
            # Test different page sizes
            page_sizes = [10, 50, 100, 500]
            results = {}

            for page_size in page_sizes:
                start = time.perf_counter()

                # Fetch first page
                messages = Message.query().order_by(
                    Message.created_at.desc()
                ).limit(page_size).all()

                # Fetch 5 more pages
                for _ in range(5):
                    if messages:
                        last_created_at = messages[-1].created_at
                        messages = Message.query().filter(
                            Message.created_at < last_created_at
                        ).order_by(Message.created_at.desc()).limit(page_size).all()

                elapsed = (time.perf_counter() - start) * 1000
                results[page_size] = elapsed

            print(f"\nPagination performance (5000 messages, 6 pages each):")
            for size, elapsed in results.items():
                print(f"  Page size {size}: {elapsed:.2f}ms")

            return results

        db_manager.with_db_session(setup_data)()
        results = db_manager.with_db_session(test_pagination)()

        # All pagination should complete reasonably fast
        for size, elapsed in results.items():
            assert elapsed < 1000, f"Pagination with size {size} too slow: {elapsed:.2f}ms"


class TestMemoryEfficiency:
    """Tests for memory efficiency with large result sets."""

    def test_streaming_vs_loading_all(self):
        """Compare memory usage between loading all results vs streaming."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class LargeRecord(ModelBase):
            __tablename__ = 'perf_large_records'
            data: Mapped[str] = mapped_column(Text)

        db_manager.base.metadata.create_all(db_manager.engine)

        def setup_data():
            # Create 1000 records with ~1KB of data each
            for i in range(1000):
                record = LargeRecord(data="x" * 1000)
                record.create()

        def test_load_all():
            import tracemalloc
            tracemalloc.start()

            records = LargeRecord.query().all()
            count = len(records)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            print(f"\nMemory usage (1000 records with 1KB data each):")
            print(f"  Records loaded: {count}")
            print(f"  Peak memory: {peak / 1024:.2f} KB")

            return peak

        db_manager.with_db_session(setup_data)()
        peak_memory = db_manager.with_db_session(test_load_all)()

        # Memory should be reasonable (not more than 10MB for this dataset)
        assert peak_memory < 10 * 1024 * 1024, f"Memory usage too high: {peak_memory / 1024 / 1024:.2f} MB"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
