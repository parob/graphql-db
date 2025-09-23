import logging
import uuid
from collections.abc import Callable
from typing import TypeVar

from context_helper import Context, ctx
from graphql_api import GraphQLAPI
from sqlalchemy import UUID, String, TypeDecorator, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
)
from sqlalchemy_utils import create_database, database_exists

from graphql_db.relay_base import RelayBase


class UUIDType(TypeDecorator):
    """SQLAlchemy UUID type using String for SQLite compatibility."""
    impl = String
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(UUID())
        else:
            return dialect.type_descriptor(String(36))

    def process_bind_param(self, value, dialect):
        if value is not None:
            if isinstance(value, uuid.UUID):
                return str(value)
            elif isinstance(value, str):
                # Validate it's a proper UUID string
                uuid.UUID(value)
                return value
        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            if isinstance(value, str):
                return uuid.UUID(value)
        return value


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class."""
    pass


T = TypeVar('T', bound='ModelBase')


@GraphQLAPI.type(abstract=True)
class ModelBase(RelayBase, Base):
    """Base model class with common functionality."""
    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        UUIDType, primary_key=True, default=uuid.uuid4
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__()

    @classmethod
    def query(cls, session: Session | None = None):
        """Get a query for this model."""
        if session is None:
            if not hasattr(ctx, 'db_session') or ctx.db_session is None:
                raise AttributeError(
                    "db_session not set in the current context"
                )
            session = ctx.db_session
        return session.query(cls)

    @classmethod
    def filter(cls, *args, session: Session | None = None, **kwargs):
        """Filter query with conditions."""
        query = cls.query(session=session)
        if args:
            query = query.filter(*args)
        if kwargs:
            query = query.filter_by(**kwargs)
        return query

    @classmethod
    def get(
        cls: type[T], id: uuid.UUID = None, session: Session | None = None
    ) -> T | None:
        """Get a single instance by ID."""
        if id:
            return cls.filter(id=id, session=session).one_or_none()
        return None

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"<{self.__class__.__name__} id: '{str(self.id)[:8]}'>"

    def create(self, session: Session | None = None) -> bool:
        """Add this instance to the session."""
        if session is None:
            session = ctx.db_session

        session.add(self)
        # Make sure object is visible to queries in same session
        session.flush()
        return True

    def delete(self, session: Session | None = None) -> bool:
        """Mark this instance for deletion."""
        if session is None:
            session = ctx.db_session

        session.delete(self)
        # Make sure deletion is visible to queries in same session
        session.flush()
        return True


class DatabaseManager:
    """Database connection and session management."""

    def __init__(
            self,
            url: str = "sqlite:///pool.db",
            install: bool = True,
            wipe: bool = False
    ):
        self.logger = logging.getLogger("db")
        self.logger.info(f"Connecting DatabaseService with url {url}")

        self.url = url
        self.base = Base
        self.engine: Engine | None = None
        self.SessionLocal: sessionmaker | None = None

        self.setup(install=install, wipe=wipe)

    def setup(self, install: bool = True, wipe: bool = False):
        """Initialize database connection and tables."""
        if install:
            if not database_exists(self.url):
                create_database(self.url)

        self.engine = create_engine(self.url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        if install:
            if wipe:
                self.logger.info(f"Wiping database at '{self.url}'")
                self.base.metadata.drop_all(self.engine)

            self.logger.info("Creating tables.")
            self.base.metadata.create_all(self.engine)

    def session(self) -> Session:
        """Create a new database session."""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized. Call setup() first.")
        return self.SessionLocal()

    def is_empty(self) -> bool:
        """Check if database has any tables."""
        if not self.engine:
            return True
        from sqlalchemy import inspect
        inspector = inspect(self.engine)
        return len(inspector.get_table_names()) == 0

    def wipe(self):
        """Drop all tables."""
        if self.engine:
            self.base.metadata.drop_all(self.engine)

    def create_all(self, base=None):
        """Create all tables."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized.")
        if base is None:
            base = self.base
        base.metadata.create_all(self.engine)

    def with_db_session(
            self,
            func: Callable = None,
            context_key_name="db_session"
    ):
        """
        Create a db session, then wrap `func`
        in a new context so it can access the db session.
        """
        def with_context(*args, **kwargs):
            db_session = self.session()
            response = None

            try:
                with Context(**{context_key_name: db_session}):
                    response = func(*args, **kwargs)

            except Exception as err:
                db_session.rollback()
                raise err
            else:
                db_session.commit()
            finally:
                db_session.close()

            return response

        return with_context
