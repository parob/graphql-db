from uuid import UUID

from graphql_api import GraphQLAPI

# noinspection DuplicatedCode
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from graphql_db.mixin import GraphQLSQLAlchemyMixin
from graphql_db.orm_base import RelayBase, UUIDType


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

    def test_relay(self):

        engine = create_engine('sqlite:///:memory:', echo=True)
        Base = declarative_base()

        class Person(RelayBase, Base):
            __tablename__ = "person"
            id = Column(UUIDType, primary_key=True)  # type: ignore
            name = Column(String)
            age = Column(Integer)

        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        session = Session()

        ed = Person(name='ed', age=55)
        session.add(ed)

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self, id: UUID) -> Person:
                return session.query(Person).filter_by(id=id).first()

        gql_query = f'''
            query GetPerson {{
                person(id: "{str(ed.id)}") {{
                    id
                    name
                    age
                }}
            }}
        '''

        result = schema.executor().execute(gql_query)

        expected = {
            "person": {
                "id": str(ed.id),
                "name": "ed",
                "age": 55
            }
        }

        assert expected == result.data
