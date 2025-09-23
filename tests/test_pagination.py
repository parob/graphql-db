"""Pagination and performance tests for graphql-db package."""
from typing import Optional

from graphql_api import GraphQLAPI
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime, timedelta

from graphql_db.orm_base import DatabaseManager, ModelBase
from graphql_db.relay_base import relay_connection


class TestPagination:
    """Test pagination, ordering, and performance features."""

    def test_basic_pagination(self):
        """Test basic limit/offset pagination."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Article(ModelBase):
            __tablename__ = 'articles'

            title: Mapped[str] = mapped_column(String(200))
            content: Mapped[str] = mapped_column(String(1000))
            view_count: Mapped[int] = mapped_column(Integer, default=0)
            created_at: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_pagination():
            # Create 50 articles
            articles_data = []
            for i in range(50):
                article = Article(
                    title=f"Article {i:02d}",
                    content=f"Content for article {i}",
                    view_count=i * 10,
                    created_at=datetime.utcnow() - timedelta(days=i)
                )
                article.create()
                articles_data.append(article)

            # Test basic queries
            all_articles = Article.query().all()
            assert len(all_articles) == 50

            # Test limit
            first_10 = Article.query().limit(10).all()
            assert len(first_10) == 10

            # Test offset
            next_10 = Article.query().offset(10).limit(10).all()
            assert len(next_10) == 10

            # Ensure different results
            first_10_ids = {a.id for a in first_10}
            next_10_ids = {a.id for a in next_10}
            assert first_10_ids.isdisjoint(next_10_ids)

            # Test ordering with pagination
            by_views_desc = Article.query().order_by(
                Article.view_count.desc()
            ).limit(5).all()
            assert len(by_views_desc) == 5
            assert by_views_desc[0].view_count >= by_views_desc[1].view_count

            # Test filtering with pagination
            high_view_articles = Article.query().filter(
                Article.view_count > 250
            ).limit(10).all()
            assert all(a.view_count > 250 for a in high_view_articles)

            # Test complex ordering
            by_date_title = Article.query().order_by(
                Article.created_at.desc(), Article.title.asc()
            ).limit(5).all()
            assert len(by_date_title) == 5

        db_manager.with_db_session(test_pagination)()

    def test_relay_style_pagination(self):
        """Test Relay-style cursor-based pagination."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Product(ModelBase):
            __tablename__ = 'products'

            name: Mapped[str] = mapped_column(String(100))
            price: Mapped[int] = mapped_column(Integer)  # Price in cents
            category: Mapped[str] = mapped_column(String(50))
            created_at: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        # Create Relay connection
        ProductConnection = relay_connection(Product)  # type: ignore

        def test_relay_pagination():
            # Create 20 products
            categories = ["electronics", "books", "clothing", "home"]
            for i in range(20):
                product = Product(
                    name=f"Product {i:02d}",
                    price=(i + 1) * 100,  # $1.00, $2.00, etc.
                    category=categories[i % len(categories)],
                    created_at=datetime.utcnow() - timedelta(hours=i)
                )
                product.create()

            # Test basic connection
            connection = ProductConnection(model=Product, first=10)
            edges = connection.edges()
            assert len(edges) <= 10

            page_info = connection.page_info()
            assert page_info is not None

            # Test filtering with connection
            from graphql_db.filter import Filter
            electronics_filter = Filter()
            electronics_filter.add_condition(Product.category == "electronics")

            electronics_connection = ProductConnection(
                model=Product,
                filter=electronics_filter,
                first=5
            )
            electronics_edges = electronics_connection.edges()
            assert all(
                edge.node.category == "electronics"
                for edge in electronics_edges
            )

            # Test ordering with connection
            from graphql_db.order_by import OrderBy, OrderByDirection
            price_order = OrderBy("price", OrderByDirection.desc)

            price_connection = ProductConnection(
                model=Product,
                order_by=[price_order],
                first=5
            )
            price_edges = price_connection.edges()
            assert len(price_edges) <= 5
            if len(price_edges) > 1:
                assert price_edges[0].node.price >= price_edges[1].node.price

        db_manager.with_db_session(test_relay_pagination)()

    def test_graphql_pagination_queries(self):
        """Test pagination through GraphQL queries."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Book(ModelBase):
            __tablename__ = 'books'

            title: Mapped[str] = mapped_column(String(200))
            author: Mapped[str] = mapped_column(String(100))
            pages: Mapped[int] = mapped_column(Integer)
            published_year: Mapped[int] = mapped_column(Integer)

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Query:
            @schema.field
            def books(
                self,
                limit: Optional[int] = 10,
                offset: Optional[int] = 0,
                order_by: Optional[str] = "title"
            ) -> list[Book]:
                query = Book.query()

                # Apply ordering
                if order_by == "title":
                    query = query.order_by(Book.title.asc())
                elif order_by == "author":
                    query = query.order_by(Book.author.asc())
                elif order_by == "pages":
                    query = query.order_by(Book.pages.desc())
                elif order_by == "year":
                    query = query.order_by(Book.published_year.desc())

                # Apply pagination
                if offset:
                    query = query.offset(offset)
                if limit:
                    query = query.limit(limit)

                return query.all()

            @schema.field
            def books_by_author(
                self, author: str, limit: Optional[int] = 10
            ) -> list[Book]:
                return Book.query().filter(
                    Book.author.like(f"%{author}%")
                ).limit(limit).all()

            @schema.field
            def book_count(self) -> int:
                return Book.query().count()

        def test_graphql_pagination():
            # Create test books
            books_data = [
                ("The Great Gatsby", "F. Scott Fitzgerald", 180, 1925),
                ("To Kill a Mockingbird", "Harper Lee", 281, 1960),
                ("1984", "George Orwell", 328, 1949),
                ("Pride and Prejudice", "Jane Austen", 279, 1813),
                ("The Catcher in the Rye", "J.D. Salinger", 277, 1951),
                ("Animal Farm", "George Orwell", 112, 1945),
                ("Lord of the Flies", "William Golding", 224, 1954),
                ("Brave New World", "Aldous Huxley", 268, 1932),
                ("The Hobbit", "J.R.R. Tolkien", 310, 1937),
                ("Fahrenheit 451", "Ray Bradbury", 249, 1953),
            ]

            for title, author, pages, year in books_data:
                book = Book(
                    title=title,
                    author=author,
                    pages=pages,
                    published_year=year
                )
                book.create()

            # Test basic pagination query - first 5 books
            first_5_query = '''
                query GetBooks {
                    books(limit: 5, offset: 0) {
                        title
                        author
                        pages
                    }
                }
            '''

            result = schema.executor().execute(first_5_query)
            assert result.data is not None
            assert len(result.data["books"]) == 5

            # Test next 5 books
            next_5_query = '''
                query GetBooks {
                    books(limit: 5, offset: 5) {
                        title
                        author
                        pages
                    }
                }
            '''

            result = schema.executor().execute(next_5_query)
            assert result.data is not None
            assert len(result.data["books"]) == 5

            # Test ordering
            ordered_query = '''
                query GetBooksByAuthor {
                    books(orderBy: "author", limit: 3) {
                        title
                        author
                    }
                }
            '''

            result = schema.executor().execute(ordered_query)
            assert result.data is not None
            authors = [book["author"] for book in result.data["books"]]
            assert authors == sorted(authors)  # Should be alphabetically
            # sorted

            # Test filtering with pagination
            author_query = '''
                query GetBooksByAuthor {
                    booksByAuthor(author: "George", limit: 5) {
                        title
                        author
                    }
                }
            '''

            result = schema.executor().execute(author_query)
            assert result.data is not None
            assert len(result.data["booksByAuthor"]) == 2  # Both Orwell books
            assert all(
                "George" in book["author"]
                for book in result.data["booksByAuthor"]
            )

            # Test count query
            count_query = '''
                query GetBookCount {
                    bookCount
                }
            '''

            result = schema.executor().execute(count_query)
            assert result.data is not None
            assert result.data["bookCount"] == 10

        db_manager.with_db_session(test_graphql_pagination)()

    def test_performance_with_large_dataset(self):
        """Test performance considerations with larger datasets."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class LogEntry(ModelBase):
            __tablename__ = 'log_entries'

            level: Mapped[str] = mapped_column(String(10))  # INFO, WARN, ERROR
            message: Mapped[str] = mapped_column(String(500))
            timestamp: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )
            source: Mapped[str] = mapped_column(String(50))

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_performance():
            import time

            # Create 1000 log entries
            levels = ["INFO", "WARN", "ERROR"]
            sources = ["app", "database", "auth", "api", "worker"]

            start_time = time.time()
            for i in range(1000):
                entry = LogEntry(
                    level=levels[i % len(levels)],
                    message=f"Log message {i}",
                    timestamp=datetime.utcnow() - timedelta(minutes=i),
                    source=sources[i % len(sources)]
                )
                entry.create()

            creation_time = time.time() - start_time
            print(f"Created 1000 entries in {creation_time:.2f} seconds")

            # Test query performance
            start_time = time.time()
            all_entries = LogEntry.query().all()
            query_time = time.time() - start_time
            print(
                f"Queried all {len(all_entries)} entries in "
                f"{query_time:.3f} seconds"
            )
            assert len(all_entries) == 1000

            # Test filtered query performance
            start_time = time.time()
            error_entries = LogEntry.query().filter(
                LogEntry.level == "ERROR"
            ).all()
            filter_time = time.time() - start_time
            print(
                f"Filtered {len(error_entries)} ERROR entries in "
                f"{filter_time:.3f} seconds"
            )

            # Test paginated query performance
            start_time = time.time()
            recent_entries = LogEntry.query().order_by(
                LogEntry.timestamp.desc()
            ).limit(50).all()
            pagination_time = time.time() - start_time
            print(
                f"Got {len(recent_entries)} recent entries in "
                f"{pagination_time:.3f} seconds"
            )
            assert len(recent_entries) == 50

            # Test complex query performance
            start_time = time.time()
            complex_query = LogEntry.query().filter(
                (LogEntry.level.in_(["ERROR", "WARN"])) &
                (LogEntry.source == "api")
            ).order_by(
                LogEntry.timestamp.desc()
            ).limit(20).all()
            complex_time = time.time() - start_time
            print(
                f"Complex query returned {len(complex_query)} entries in "
                f"{complex_time:.3f} seconds"
            )

            # Verify results
            assert all(
                entry.level in ["ERROR", "WARN"] for entry in complex_query
            )
            assert all(entry.source == "api" for entry in complex_query)

        db_manager.with_db_session(test_performance)()
