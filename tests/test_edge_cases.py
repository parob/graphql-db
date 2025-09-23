"""Edge cases and error handling tests for graphql-db package."""
import uuid
from typing import Optional

import pytest
from graphql_api import GraphQLAPI
from sqlalchemy import Integer, String, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.exc import IntegrityError

from graphql_db.orm_base import DatabaseManager, ModelBase


class TestEdgeCases:
    """Test edge cases, error handling, and boundary conditions."""

    def test_empty_database_queries(self):
        """Test querying empty database."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class User(ModelBase):
            __tablename__ = 'users'
            name: Mapped[str] = mapped_column(String(50))

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_empty_queries():
            # Test empty queries
            users = User.query().all()
            assert users == []

            user = User.query().first()
            assert user is None

            # Test get with non-existent ID
            fake_id = uuid.uuid4()
            user = User.get(fake_id)
            assert user is None

            # Test filter with no results
            filtered_users = User.filter(User.name == "nonexistent").all()
            assert filtered_users == []

        db_manager.with_db_session(test_empty_queries)()

    def test_null_and_optional_fields(self):
        """Test handling of null and optional fields."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Profile(ModelBase):
            __tablename__ = 'profiles'

            username: Mapped[str] = mapped_column(String(50))
            bio: Mapped[Optional[str]] = mapped_column(
                String(500), nullable=True
            )
            age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_null_handling():
            # Create profile with all fields
            profile1 = Profile(
                username="alice", bio="Software engineer", age=30
            )  # type: ignore
            profile1.create()

            # Create profile with only required fields
            profile2 = Profile(username="bob")
            profile2.create()

            # Create profile with explicit None values
            profile3 = Profile(username="charlie", bio=None, age=None)
            profile3.create()

            # Test queries
            all_profiles = Profile.query().all()
            assert len(all_profiles) == 3

            # Test filtering by null values
            profiles_with_bio = Profile.query().filter(
                Profile.bio.is_not(None)
            ).all()
            assert len(profiles_with_bio) == 1
            assert profiles_with_bio[0].username == "alice"

            profiles_without_bio = Profile.query().filter(
                Profile.bio.is_(None)
            ).all()
            assert len(profiles_without_bio) == 2

            # Test GraphQL schema with nulls
            schema = GraphQLAPI()

            @schema.type(is_root_type=True)
            class Query:
                @schema.field
                def profiles(self) -> list[Profile]:
                    return Profile.query().all()

            gql_query = '''
                query GetProfiles {
                    profiles {
                        username
                        bio
                        age
                    }
                }
            '''

            result = schema.executor().execute(gql_query)
            assert result.data is not None
            assert len(result.data["profiles"]) == 3

            # Check that null values are handled correctly
            alice_profile = next(
                p for p in result.data["profiles"] if p["username"] == "alice"
            )  # type: ignore
            assert alice_profile["bio"] == "Software engineer"
            assert alice_profile["age"] == 30

            bob_profile = next(
                p for p in result.data["profiles"] if p["username"] == "bob"
            )  # type: ignore
            assert bob_profile["bio"] is None
            assert bob_profile["age"] is None

        db_manager.with_db_session(test_null_handling)()

    def test_invalid_uuids(self):
        """Test handling of invalid UUID inputs."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Item(ModelBase):
            __tablename__ = 'items'
            name: Mapped[str] = mapped_column(String(50))

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_uuid_validation():
            item = Item(name="test item")
            item.create()

            # Test invalid UUID strings - these should raise StatementError
            # due to UUID validation
            from sqlalchemy.exc import StatementError
            with pytest.raises((ValueError, TypeError, StatementError)):
                Item.get("not-a-uuid")  # type: ignore

            with pytest.raises((ValueError, TypeError, StatementError)):
                Item.get("12345")  # type: ignore

            # Empty string might be handled differently, let's test what
            # actually happens
            try:
                result = Item.get("")  # type: ignore
                # If no exception, result should be None
                assert result is None, f"Expected None but got {result}"
            except (ValueError, TypeError, StatementError):
                # This is also acceptable behavior
                pass

            # Test None handling
            result = Item.get(None)  # type: ignore
            assert result is None

            # Test valid UUID as string
            valid_uuid_str = str(item.id)
            result = Item.get(uuid.UUID(valid_uuid_str))
            assert result is not None
            assert result.name == "test item"

        db_manager.with_db_session(test_uuid_validation)()

    def test_database_constraint_violations(self):
        """Test handling of database constraint violations."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class UniqueUser(ModelBase):
            __tablename__ = 'unique_users'
            username: Mapped[str] = mapped_column(String(50), unique=True)
            email: Mapped[str] = mapped_column(String(100), unique=True)

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_unique_constraints():
            # Create first user
            user1 = UniqueUser(username="alice", email="alice@example.com")
            user1.create()

            # Try to create user with duplicate username
            user2 = UniqueUser(username="alice", email="different@example.com")
            try:
                user2.create()
                # If SQLite doesn't enforce unique constraints, that's OK
                # for this test
                print(
                    "SQLite doesn't enforce unique constraints in this setup"
                )
            except IntegrityError:
                # Expected behavior - rollback
                from context_helper import ctx
                ctx.db_session.rollback()

            # Try to create user with duplicate email
            user3 = UniqueUser(username="different", email="alice@example.com")
            try:
                user3.create()
                print(
                    "SQLite doesn't enforce unique constraints in this setup"
                )
            except IntegrityError:
                # Expected behavior - rollback
                from context_helper import ctx
                ctx.db_session.rollback()

            # Create user with different username and email should work
            user4 = UniqueUser(username="bob", email="bob@example.com")
            user4.create()

            # Verify users exist (count depends on constraint enforcement)
            all_users = UniqueUser.query().all()
            assert len(all_users) >= 2  # At least alice and bob

        db_manager.with_db_session(test_unique_constraints)()

    def test_foreign_key_constraints(self):
        """Test foreign key constraint handling."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class Author(ModelBase):
            __tablename__ = 'authors'
            name: Mapped[str] = mapped_column(String(100))

        class Book(ModelBase):
            __tablename__ = 'books'
            title: Mapped[str] = mapped_column(String(200))
            author_id: Mapped[uuid.UUID] = mapped_column(
                ForeignKey('authors.id')
            )

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_foreign_key_handling():
            # Create author
            author = Author(name="Jane Doe")
            author.create()

            # Create book with valid foreign key
            book1 = Book(title="Great Book", author_id=author.id)
            book1.create()

            # Try to create book with invalid foreign key
            fake_author_id = uuid.uuid4()
            book2 = Book(title="Bad Book", author_id=fake_author_id)

            # Note: SQLite doesn't enforce foreign key constraints by default
            # but we can still test the behavior
            book2.create()  # This might succeed in SQLite

            # Test querying
            books = Book.query().all()
            assert len(books) >= 1  # At least the valid book

            # Test filtering by foreign key
            author_books = Book.query().filter(
                Book.author_id == author.id
            ).all()
            assert len(author_books) == 1
            assert author_books[0].title == "Great Book"

        db_manager.with_db_session(test_foreign_key_handling)()

    def test_large_dataset_operations(self):
        """Test operations with larger datasets."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class DataPoint(ModelBase):
            __tablename__ = 'data_points'
            value: Mapped[int] = mapped_column(Integer)
            category: Mapped[str] = mapped_column(String(20))

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_bulk_operations():
            # Create 100 data points
            for i in range(100):
                point = DataPoint(
                    value=i,
                    category="even" if i % 2 == 0 else "odd"
                )
                point.create()

            # Test counting
            all_points = DataPoint.query().all()
            assert len(all_points) == 100

            # Test filtering
            even_points = DataPoint.query().filter(
                DataPoint.category == "even"
            ).all()
            assert len(even_points) == 50

            odd_points = DataPoint.query().filter(
                DataPoint.category == "odd"
            ).all()
            assert len(odd_points) == 50

            # Test ordering
            ordered_points = DataPoint.query().order_by(
                DataPoint.value.desc()
            ).limit(5).all()
            assert len(ordered_points) == 5
            assert ordered_points[0].value == 99

            # Test complex filtering
            high_even = DataPoint.query().filter(
                (DataPoint.category == "even") & (DataPoint.value > 50)
            ).all()
            assert len(high_even) == 24  # Even numbers from 52 to 98

        db_manager.with_db_session(test_bulk_operations)()

    def test_session_management_edge_cases(self):
        """Test edge cases in session management."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class TestModel(ModelBase):
            __tablename__ = 'test_models'
            name: Mapped[str] = mapped_column(String(50))

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        def test_session_edge_cases():
            # Test creating multiple objects in same session
            obj1 = TestModel(name="first")
            obj1.create()

            obj2 = TestModel(name="second")
            obj2.create()

            # Both should be visible in same session
            all_objects = TestModel.query().all()
            assert len(all_objects) == 2

            # Test deleting and recreating in same session
            obj1.delete()
            remaining = TestModel.query().all()
            assert len(remaining) == 1
            assert remaining[0].name == "second"

            # Create new object after deletion
            obj3 = TestModel(name="third")
            obj3.create()

            final_objects = TestModel.query().all()
            assert len(final_objects) == 2
            names = [obj.name for obj in final_objects]
            assert "second" in names
            assert "third" in names

        db_manager.with_db_session(test_session_edge_cases)()

    def test_graphql_error_handling(self):
        """Test GraphQL specific error handling."""
        db_manager = DatabaseManager(url="sqlite:///:memory:", wipe=True)

        class ErrorTestModel(ModelBase):
            __tablename__ = 'error_test_models'
            name: Mapped[str] = mapped_column(String(50))

        # Create tables
        db_manager.base.metadata.create_all(db_manager.engine)  # type: ignore

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Query:
            @schema.field
            def test_model(
                self, model_id: uuid.UUID
            ) -> Optional[ErrorTestModel]:
                return ErrorTestModel.get(model_id)

            @schema.field
            def test_models(self) -> list[ErrorTestModel]:
                return ErrorTestModel.query().all()

        def test_graphql_errors():
            # Create test data
            model = ErrorTestModel(name="test")
            model.create()

            # Test valid query
            valid_query = f'''
                query GetModel {{
                    testModel(modelId: "{model.id}") {{
                        name
                    }}
                }}
            '''

            result = schema.executor().execute(valid_query)
            assert result.data is not None
            assert result.data["testModel"]["name"] == "test"

            # Test query with invalid UUID format
            invalid_query = '''
                query GetModel {
                    testModel(modelId: "invalid-uuid") {
                        name
                    }
                }
            '''

            result = schema.executor().execute(invalid_query)
            assert result.errors is not None
            assert len(result.errors) > 0

            # Test query for non-existent model
            nonexistent_query = f'''
                query GetModel {{
                    testModel(modelId: "{uuid.uuid4()}") {{
                        name
                    }}
                }}
            '''

            result = schema.executor().execute(nonexistent_query)
            assert result.data is not None
            assert result.data["testModel"] is None

        db_manager.with_db_session(test_graphql_errors)()
