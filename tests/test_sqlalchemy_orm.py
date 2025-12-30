from typing import Optional
from graphql_api import GraphQLAPI
from sqlalchemy import Integer, String
from sqlalchemy.ext.hybrid import hybrid_property
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


class PersonWithHybrid(ModelBase):
    __tablename__ = 'person_with_hybrid'

    first_name: Mapped[str | None] = mapped_column(String(50))
    last_name: Mapped[str | None] = mapped_column(String(50))

    def __init__(
        self,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.first_name = first_name
        self.last_name = last_name

    @hybrid_property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


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

    def test_hybrid_property(self):

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> PersonWithHybrid:
                return PersonWithHybrid(first_name="John", last_name="Doe")

        gql_query = '''
            query GetPerson {
                person {
                    firstName
                    lastName
                    fullName
                }
            }
        '''

        result = schema.executor().execute(gql_query)

        expected = {
            "person": {
                "firstName": "John",
                "lastName": "Doe",
                "fullName": "John Doe"
            }
        }

        assert expected == result.data
