from typing import Optional
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from graphql_db.orm_base import DatabaseManager, ModelBase


class Person(ModelBase):
    __tablename__ = 'person'

    name: Mapped[str | None] = mapped_column(String(50))
    age: Mapped[int | None] = mapped_column(Integer)

    def __init__(
        self,
        name: Optional[str] = None,
        age: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.name = name
        self.age = age


# noinspection DuplicatedCode
class TestModel:

    def test_create(self):
        db_manager = DatabaseManager(wipe=True)

        def create_person(expected_people_count):
            person = Person(name="rob", age=26)
            person.create()

            all_people = Person.query().all()
            assert len(all_people) == expected_people_count

        db_manager.with_db_session(create_person)(1)

        db_manager.setup()

        db_manager.with_db_session(create_person)(2)

        db_manager.setup(wipe=True)

        db_manager.with_db_session(create_person)(1)
