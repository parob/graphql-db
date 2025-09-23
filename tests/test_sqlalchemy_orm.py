from typing import Optional
from graphql_api import GraphQLAPI
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from graphql_db.orm_base import DatabaseManager, ModelBase


class Individual(ModelBase):
    __tablename__ = 'individual'

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

        def create_person():
            person = Individual(name="rob", age=26)
            person.create()

            all_people = Individual.query().all()
            assert len(all_people) == 1
            assert all_people == [person]

        db_manager.with_db_session(create_person)()

    def test_delete(self):
        db_manager = DatabaseManager(wipe=True)

        def delete_person():
            person = Individual(name="rob", age=26)
            person.create()

            all_people = Individual.query().all()
            assert len(all_people) == 1
            assert all_people == [person]

            person.delete()

            all_people = Individual.query().all()
            assert len(all_people) == 0

        db_manager.with_db_session(delete_person)()

    def test_filter(self):
        db_manager = DatabaseManager()

        def delete_person():
            person = Individual(name="rob", age=26)
            person.create()

            all_people = Individual.query().all()
            assert len(all_people) == 1
            assert all_people == [person]

            person.delete()

            all_people = Individual.query().all()
            assert len(all_people) == 0

        db_manager.with_db_session(delete_person)()

    def test_schema(self):

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> Individual:
                return Individual(name="rob", age=26)

        gql_query = '''
            query GetPerson {
                person {
                    name
                    age
                }
            }
        '''

        result = schema.executor().execute(gql_query)

        expected = {
            "person": {
                "name": "rob",
                "age": 26
            }
        }

        assert expected == result.data
