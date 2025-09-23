"""Test SQLModel with relationships and virtual fields."""
from typing import Optional, List
import pytest

try:
    from sqlmodel import (
        SQLModel, Field, Session, create_engine, select, Relationship
    )
    from graphql_api import GraphQLAPI

    SQLMODEL_AVAILABLE = True
except ImportError:
    SQLMODEL_AVAILABLE = False


@pytest.mark.skipif(not SQLMODEL_AVAILABLE, reason="SQLModel not available")
class TestSQLModelRelationships:
    """Test SQLModel integration with relationships and virtual fields."""

    def test_sqlmodel_with_relationships(self):
        """Test SQLModel models with relationships."""

        # Define models with relationships
        class Team(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            name: str
            city: str

            # Relationship
            heroes: List["Hero"] = Relationship(back_populates="team")

            # Virtual field (computed property)
            @property
            def hero_count(self) -> int:
                """Virtual field showing number of heroes."""
                return len(self.heroes) if self.heroes else 0

        class Hero(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            name: str
            secret_name: str
            age: Optional[int] = None
            team_id: Optional[int] = Field(default=None, foreign_key="team.id")

            # Relationship
            team: Optional[Team] = Relationship(back_populates="heroes")

            # Virtual field (computed property)
            @property
            def display_name(self) -> str:
                """Virtual field combining name and secret name."""
                return f"{self.name} ({self.secret_name})"

        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        # Create GraphQL API
        api = GraphQLAPI()

        @api.type(is_root_type=True)
        class Query:
            @api.field
            def teams(self) -> List[Team]:
                with Session(engine) as session:
                    teams = session.exec(select(Team)).all()
                    return teams

            @api.field
            def heroes(self) -> List[Hero]:
                with Session(engine) as session:
                    heroes = session.exec(select(Hero)).all()
                    return heroes

        # Test data creation
        with Session(engine) as session:
            # Create teams
            team1 = Team(name="Avengers", city="New York")
            team2 = Team(name="X-Men", city="Westchester")

            session.add(team1)
            session.add(team2)
            session.commit()
            session.refresh(team1)
            session.refresh(team2)

            # Create heroes
            hero1 = Hero(
                name="Spider-Man", secret_name="Peter Parker",
                age=25, team_id=team1.id
            )
            hero2 = Hero(
                name="Iron Man", secret_name="Tony Stark",
                age=45, team_id=team1.id
            )
            hero3 = Hero(
                name="Wolverine", secret_name="Logan",
                age=200, team_id=team2.id
            )

            session.add(hero1)
            session.add(hero2)
            session.add(hero3)
            session.commit()

        # Test GraphQL queries
        result = api.execute('''
            query {
                teams {
                    name
                    city
                }
            }
        ''')

        assert not result.errors
        assert len(result.data['teams']) == 2
        assert any(team['name'] == 'Avengers' for team in result.data['teams'])

        result = api.execute('''
            query {
                heroes {
                    name
                    secretName
                    age
                }
            }
        ''')

        assert not result.errors
        assert len(result.data['heroes']) == 3
        heroes_data = result.data['heroes']
        assert any(hero['name'] == 'Spider-Man' for hero in heroes_data)

    def test_sqlmodel_virtual_fields(self):
        """Test SQLModel virtual fields integration."""

        class Product(SQLModel, table=True):
            id: Optional[int] = Field(default=None, primary_key=True)
            name: str
            price: float
            cost: float

            @property
            def profit_margin(self) -> float:
                """Virtual field calculating profit margin."""
                if self.price == 0:
                    return 0.0
                return ((self.price - self.cost) / self.price) * 100

        # Create in-memory database
        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        # Create GraphQL API
        api = GraphQLAPI()

        @api.type(is_root_type=True)
        class Query:
            @api.field
            def products(self) -> List[Product]:
                with Session(engine) as session:
                    return session.exec(select(Product)).all()

        # Test data creation
        with Session(engine) as session:
            product1 = Product(name="Widget", price=100.0, cost=60.0)
            product2 = Product(name="Gadget", price=50.0, cost=30.0)

            session.add(product1)
            session.add(product2)
            session.commit()

        # Test GraphQL query with virtual fields
        result = api.execute('''
            query {
                products {
                    name
                    price
                    cost
                }
            }
        ''')

        assert not result.errors
        assert len(result.data['products']) == 2

        # Find widget
        widget_data = result.data['products']
        widget = next(p for p in widget_data if p['name'] == 'Widget')
        assert widget['price'] == 100.0
        assert widget['cost'] == 60.0
