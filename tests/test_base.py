import uuid

from graphql_api import GraphQLAPI

# noinspection DuplicatedCode
from sqlalchemy import UUID, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

from graphql_db.mixin import GraphQLSQLAlchemyMixin
from graphql_db.orm_base import UUIDType


class TestModel:

    def test_basic(self):
        Base = declarative_base()

        from sqlalchemy import Column, Integer, String

        class Person(GraphQLSQLAlchemyMixin, Base):
            __tablename__ = 'people'
            id = Column(Integer, primary_key=True)
            name = Column(String)
            age = Column(Integer)

        ed = Person(name='ed', age=55)  # type: ignore

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> Person:
                return ed

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
                "name": "ed",
                "age": 55
            }
        }

        assert expected == result.data

    def test_uuid(self):
        Base = declarative_base()

        from sqlalchemy import Column, String

        class Person(GraphQLSQLAlchemyMixin, Base):
            __tablename__ = 'people'
            id = Column(
                UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
            )
            name = Column(String)

        person_id = uuid.uuid4()

        person = Person(id=person_id, name='joe')  # type: ignore

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> Person:
                return person

        gql_query = '''
                query GetPerson {
                    person {
                        id
                        name
                    }
                }
            '''

        result = schema.executor().execute(gql_query)
        assert result.data is not None
        assert person_id == uuid.UUID(result.data["person"]["id"])

    def test_relationship(self):
        Base = declarative_base()

        from sqlalchemy import Column, Integer, String

        class Person(GraphQLSQLAlchemyMixin, Base):
            __tablename__ = 'people'
            id = Column(Integer, primary_key=True)
            name = Column(String)
            age = Column(Integer)
            home_id = Column(UUIDType, ForeignKey("property.id"))

            home = relationship(
                "Property",
                remote_side="Property.id",
                foreign_keys=home_id,
                back_populates="home_inhabitants",
            )

            holiday_home_id = Column(
                UUIDType, ForeignKey("property.id"), nullable=True
            )

            holiday_home = relationship(
                "Property",
                remote_side="Property.id",
                foreign_keys=holiday_home_id,
                back_populates="holiday_inhabitants",
            )

        class Property(GraphQLSQLAlchemyMixin, Base):
            __tablename__ = 'property'
            id = Column(Integer, primary_key=True)
            name = Column(String)

            home_inhabitants = relationship(
                "Person",
                cascade="all, delete-orphan, save-update",
                foreign_keys=Person.home_id,
                back_populates="home",
            )

            holiday_inhabitants = relationship(
                "Person",
                cascade="all, delete-orphan, save-update",
                foreign_keys=Person.holiday_home_id,
                back_populates="holiday_home",
            )

        steve = Person(name='steve', age=55)  # type: ignore
        steve.home = Property(name="steves house")  # type: ignore

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> Person:
                return steve

        gql_query = '''
            query GetPerson {
                person {
                    name
                    age
                    home {
                        name
                        homeInhabitants {
                            name
                        }
                        holidayInhabitants {
                            name
                        }
                    }
                    holidayHome {
                        name
                        homeInhabitants {
                            name
                        }
                        holidayInhabitants {
                            name
                        }
                    }
                }
            }
        '''

        result = schema.executor().execute(gql_query)

        expected = {
            "person": {
                "name": "steve",
                "age": 55,
                "home": {
                    "name": "steves house",
                    "homeInhabitants": [{'name': 'steve'}],
                    'holidayInhabitants': []
                },
                'holidayHome': None
            }
        }

        assert expected == result.data

        graphql_schema = schema.build_schema()

        assert graphql_schema is not None
        assert graphql_schema[0].query_type is not None
        assert graphql_schema[0].query_type.fields is not None
        person_fields = graphql_schema[0].query_type.fields["person"]
        assert person_fields is not None
        assert person_fields.type.of_type.fields
