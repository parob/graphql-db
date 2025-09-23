"""Comprehensive integration tests for graphql-db package."""
import uuid
from typing import Optional

from graphql_api import GraphQLAPI
from sqlalchemy import Integer, String, ForeignKey, DateTime, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime

from graphql_db.orm_base import DatabaseManager, ModelBase


class TestIntegration:
    """Test complete GraphQL API integration with database models."""

    def test_complete_blog_api(self):
        """Test a complete blog API with Users, Posts, and Comments."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        # Define related models
        class User(ModelBase):
            __tablename__ = 'users'

            username: Mapped[str] = mapped_column(String(50), unique=True)
            email: Mapped[str] = mapped_column(String(100), unique=True)
            is_active: Mapped[bool] = mapped_column(Boolean, default=True)
            created_at: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )

            # Relationships
            posts = relationship("Post", back_populates="author")
            comments = relationship("Comment", back_populates="author")

        class Post(ModelBase):
            __tablename__ = 'posts'

            title: Mapped[str] = mapped_column(String(200))
            content: Mapped[str] = mapped_column(Text)
            published: Mapped[bool] = mapped_column(Boolean, default=False)
            created_at: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )
            author_id: Mapped[uuid.UUID] = mapped_column(
                ForeignKey('users.id')
            )

            # Relationships
            author = relationship("User", back_populates="posts")
            comments = relationship("Comment", back_populates="post")

        class Comment(ModelBase):
            __tablename__ = 'comments'

            content: Mapped[str] = mapped_column(Text)
            created_at: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )
            author_id: Mapped[uuid.UUID] = mapped_column(
                ForeignKey('users.id')
            )
            post_id: Mapped[uuid.UUID] = mapped_column(
                ForeignKey('posts.id')
            )

            # Relationships
            author = relationship("User", back_populates="comments")
            post = relationship("Post", back_populates="comments")

        # Manually create tables since models are defined locally
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        # Create GraphQL API
        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Query:
            @schema.field
            def users(self) -> list[User]:
                return User.query().all()

            @schema.field
            def posts(self) -> list[Post]:
                return Post.query().filter(Post.published.is_(True)).all()

            @schema.field
            def user(self, username: str) -> Optional[User]:
                return User.query().filter(User.username == username).first()

        @schema.type
        class Mutation:
            @schema.field
            def create_user(self, username: str, email: str) -> User:
                user = User(username=username, email=email)
                user.create()
                return user

            @schema.field
            def create_post(
                self, title: str, content: str, author_id: uuid.UUID
            ) -> Post:
                post = Post(title=title, content=content, author_id=author_id)
                post.create()
                return post

            @schema.field
            def publish_post(self, post_id: uuid.UUID) -> Optional[Post]:
                post = Post.get(post_id)
                if post:
                    post.published = True
                    post.create()  # This will update the existing record
                return post

        def test_blog_operations():
            # Create users
            user1 = User(username="alice", email="alice@example.com")
            user1.create()

            user2 = User(username="bob", email="bob@example.com")
            user2.create()

            # Create posts
            post1 = Post(
                title="My First Post",
                content="This is my first blog post!",
                author_id=user1.id,
                published=True
            )
            post1.create()

            post2 = Post(
                title="Draft Post",
                content="This is a draft",
                author_id=user1.id,
                published=False
            )
            post2.create()

            # Create comments
            comment1 = Comment(
                content="Great post!",
                author_id=user2.id,
                post_id=post1.id
            )
            comment1.create()

            # Test queries
            all_users = User.query().all()
            assert len(all_users) == 2
            assert all_users[0].username in ["alice", "bob"]

            published_posts = Post.query().filter(
                Post.published.is_(True)
            ).all()
            assert len(published_posts) == 1
            assert published_posts[0].title == "My First Post"

            # Test relationships
            alice = User.query().filter(User.username == "alice").first()
            assert alice is not None
            assert len(alice.posts) == 2
            assert alice.posts[0].author.username == "alice"

            # Test GraphQL schema generation
            gql_query = '''
                query GetUsers {
                    users {
                        username
                        email
                        isActive
                    }
                }
            '''

            result = schema.executor().execute(gql_query)
            assert result.data is not None
            assert len(result.data["users"]) == 2

        db_manager.with_db_session(test_blog_operations)()

    def test_complex_relationships(self):
        """Test complex many-to-many relationships."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Student(ModelBase):
            __tablename__ = 'students'

            name: Mapped[str] = mapped_column(String(100))
            email: Mapped[str] = mapped_column(String(100))

        class Course(ModelBase):
            __tablename__ = 'courses'

            name: Mapped[str] = mapped_column(String(100))
            description: Mapped[Optional[str]] = mapped_column(Text)

        class Enrollment(ModelBase):
            __tablename__ = 'enrollments'

            student_id: Mapped[uuid.UUID] = mapped_column(
                ForeignKey('students.id')
            )
            course_id: Mapped[uuid.UUID] = mapped_column(
                ForeignKey('courses.id')
            )
            grade: Mapped[Optional[str]] = mapped_column(String(2))
            enrolled_at: Mapped[datetime] = mapped_column(
                DateTime, default=datetime.utcnow
            )

            # Relationships
            student = relationship("Student")
            course = relationship("Course")

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_enrollment_system():
            # Create students
            alice = Student(
                name="Alice Johnson", email="alice@school.edu"
            )
            alice.create()

            bob = Student(name="Bob Smith", email="bob@school.edu")
            bob.create()

            # Create courses
            math = Course(
                name="Mathematics 101", description="Basic mathematics"
            )
            math.create()

            cs = Course(
                name="Computer Science 101", description="Intro to programming"
            )
            cs.create()

            # Create enrollments
            enrollment1 = Enrollment(
                student_id=alice.id, course_id=math.id, grade="A"
            )
            enrollment1.create()

            enrollment2 = Enrollment(
                student_id=alice.id, course_id=cs.id, grade="B+"
            )
            enrollment2.create()

            enrollment3 = Enrollment(
                student_id=bob.id, course_id=cs.id, grade="A-"
            )
            enrollment3.create()

            # Test complex queries
            enrollments = Enrollment.query().all()
            assert len(enrollments) == 3

            # Test filtering by grade
            a_grades = Enrollment.query().filter(
                Enrollment.grade.like("A%")
            ).all()
            assert len(a_grades) == 2

            # Test relationship access
            alice_enrollments = Enrollment.query().filter(
                Enrollment.student_id == alice.id
            ).all()
            assert len(alice_enrollments) == 2
            assert alice_enrollments[0].student.name == "Alice Johnson"

        db_manager.with_db_session(test_enrollment_system)()

    def test_graphql_schema_introspection(self):
        """Test GraphQL schema introspection with complex types."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Product(ModelBase):
            __tablename__ = 'products'

            name: Mapped[str] = mapped_column(String(100))
            price: Mapped[int] = mapped_column(Integer)  # Price in cents
            in_stock: Mapped[bool] = mapped_column(Boolean, default=True)
            description: Mapped[Optional[str]] = mapped_column(Text)

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        # Create GraphQL API
        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Query:
            @schema.field
            def products(self) -> list[Product]:
                return Product.query().all()

            @schema.field
            def product(self, product_id: uuid.UUID) -> Optional[Product]:
                return Product.get(product_id)

        def test_introspection():
            # Create sample data
            laptop = Product(
                name="Gaming Laptop",
                price=129999,  # $1299.99
                in_stock=True,
                description="High-performance gaming laptop"
            )
            laptop.create()

            mouse = Product(
                name="Wireless Mouse",
                price=2999,  # $29.99
                in_stock=False
            )
            mouse.create()

            # Test GraphQL introspection query
            introspection_query = '''
                query IntrospectionQuery {
                    __schema {
                        types {
                            name
                            fields {
                                name
                                type {
                                    name
                                    kind
                                }
                            }
                        }
                    }
                }
            '''

            result = schema.executor().execute(introspection_query)
            assert result.data is not None
            assert "__schema" in result.data

            # Test that Product type is in the schema
            types = result.data["__schema"]["types"]
            product_type = next(
                (t for t in types if t["name"] == "Product"), None
            )
            assert product_type is not None

            # Test field queries
            products_query = '''
                query GetProducts {
                    products {
                        name
                        price
                        inStock
                        description
                    }
                }
            '''

            result = schema.executor().execute(products_query)
            assert result.data is not None
            assert len(result.data["products"]) == 2

            # Test filtering with variables
            product_query = f'''
                query GetProduct {{
                    product(productId: "{laptop.id}") {{
                        name
                        price
                        inStock
                    }}
                }}
            '''

            result = schema.executor().execute(product_query)
            assert result.data is not None
            assert result.data["product"]["name"] == "Gaming Laptop"
            assert result.data["product"]["price"] == 129999

        db_manager.with_db_session(test_introspection)()
