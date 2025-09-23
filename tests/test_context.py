from context_helper import Context
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from graphql_db import DatabaseManager, ModelBase


class TestContent:

    def test_content(self):
        db_manager = DatabaseManager(wipe=True)

        class Thing(ModelBase):
            __tablename__ = 'thing'

            name: Mapped[str] = mapped_column(String(50))
            age: Mapped[int] = mapped_column(Integer)

        # Create the table (it should be auto-created by DatabaseManager)
        db_manager.create_all()

        db_session = db_manager.session()

        try:
            with Context(db_session=db_session):
                thing = Thing(name="rob", age=10)
                thing.create()

                # Make sure it's in the database for testing
                db_session.flush()
        finally:
            db_session.rollback()
            db_session.close()
