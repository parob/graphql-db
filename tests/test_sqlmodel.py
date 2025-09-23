from graphql_api import GraphQLAPI
from typing import Optional, List

from graphql_db import GraphQLSQLAlchemyMixin


class TestSQLModel:

    def test_sqlmodel_basic(self):

        from sqlmodel import Field, Session, SQLModel, create_engine, select

        class Hero(GraphQLSQLAlchemyMixin, SQLModel, table=True):
            id: int | None = Field(default=None, primary_key=True)
            name: str
            real_name: str
            age: int | None = None

        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        spiderman = Hero(name="Spiderman", real_name="Peter Parker")
        batman = Hero(name="Batman", real_name="Bruce Wayne")

        with Session(engine) as session:
            session.add(spiderman)
            session.add(batman)

            session.commit()
            session.close()

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def hero(self, name: str) -> Hero | None:
                with Session(engine) as session:
                    statement = select(Hero).where(Hero.name == name)
                    return session.exec(statement).first()

        gql_query = '''
                query GetBatman {
                    hero(name:"Batman") {
                        realName
                    }
                }
            '''

        result = schema.executor().execute(gql_query)
        assert result.data is not None
        assert "Bruce Wayne" == result.data["hero"]["realName"]

    def test_sqlmodel_relationships(self):
        """Test SQLModel with relationships between Team and SuperHero."""
        from sqlmodel import (
            Field, Session, SQLModel, create_engine, select, Relationship
        )

        class Team(GraphQLSQLAlchemyMixin, SQLModel, table=True):
            id: int | None = Field(default=None, primary_key=True)
            name: str
            headquarters: str

            # Relationship
            heroes: List["SuperHero"] = Relationship(back_populates="team")

        class SuperHero(GraphQLSQLAlchemyMixin, SQLModel, table=True):
            id: int | None = Field(default=None, primary_key=True)
            name: str
            real_name: str
            age: int | None = None
            team_id: int | None = Field(default=None, foreign_key="team.id")

            # Relationship
            team: Optional[Team] = Relationship(back_populates="heroes")

        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        with Session(engine) as session:
            # Create teams
            avengers = Team(name="Avengers", headquarters="Avengers Tower")
            xmen = Team(name="X-Men", headquarters="Xavier's School")
            session.add(avengers)
            session.add(xmen)
            session.commit()

            # Create heroes with team relationships
            iron_man = SuperHero(
                name="Iron Man", real_name="Tony Stark",
                age=45, team_id=avengers.id
            )
            captain_america = SuperHero(
                name="Captain America", real_name="Steve Rogers",
                age=100, team_id=avengers.id
            )
            wolverine = SuperHero(
                name="Wolverine", real_name="Logan",
                age=200, team_id=xmen.id
            )
            # No team
            solo_hero = SuperHero(
                name="Spider-Man", real_name="Peter Parker", age=25
            )

            session.add_all([iron_man, captain_america, wolverine, solo_hero])
            session.commit()
            session.close()

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Query:
            @schema.field
            def teams(self) -> List[Team]:
                with Session(engine) as session:
                    statement = select(Team)
                    return list(session.exec(statement).all())

            @schema.field
            def heroes(self) -> List[SuperHero]:
                from sqlalchemy.orm import selectinload
                with Session(engine) as session:
                    statement = select(SuperHero).options(
                        selectinload(SuperHero.team)  # type: ignore
                    )
                    return list(session.exec(statement).all())

            @schema.field
            def team_by_name(self, name: str) -> Optional[Team]:
                from sqlalchemy.orm import selectinload
                with Session(engine) as session:
                    statement = select(Team).options(selectinload(
                        Team.heroes  # type: ignore
                    )).where(Team.name == name)
                    return session.exec(statement).one_or_none()

        # Test basic team query
        team_query = '''
            query GetTeams {
                teams {
                    name
                    headquarters
                }
            }
        '''

        result = schema.executor().execute(team_query)
        assert result.data is not None
        teams = result.data["teams"]
        assert len(teams) == 2
        team_names = [team["name"] for team in teams]
        assert "Avengers" in team_names
        assert "X-Men" in team_names

        # Test hero query with team relationships
        heroes_query = '''
            query GetHeroes {
                heroes {
                    name
                    realName
                    age
                    team {
                        name
                        headquarters
                    }
                }
            }
        '''

        result = schema.executor().execute(heroes_query)
        assert result.data is not None
        heroes = result.data["heroes"]
        assert len(heroes) == 4

        # Find Iron Man and check his team
        iron_man_data = next(h for h in heroes if h["name"] == "Iron Man")
        assert iron_man_data["realName"] == "Tony Stark"
        assert iron_man_data["team"]["name"] == "Avengers"
        assert iron_man_data["team"]["headquarters"] == "Avengers Tower"

        # Find Spider-Man and check he has no team
        spiderman_data = next(h for h in heroes if h["name"] == "Spider-Man")
        assert spiderman_data["team"] is None

        # Test team with heroes relationship
        avengers_query = '''
            query GetAvengers {
                teamByName(name: "Avengers") {
                    name
                    headquarters
                    heroes {
                        name
                        realName
                    }
                }
            }
        '''

        result = schema.executor().execute(avengers_query)
        assert result.data is not None
        avengers_data = result.data["teamByName"]
        assert avengers_data["name"] == "Avengers"
        assert len(avengers_data["heroes"]) == 2
        hero_names = [hero["name"] for hero in avengers_data["heroes"]]
        assert "Iron Man" in hero_names
        assert "Captain America" in hero_names
