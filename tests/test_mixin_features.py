"""Tests for SQLAlchemy mixin features: hybrid properties, association proxies, etc."""
import enum
from typing import Optional, List
from graphql_api import GraphQLAPI
from sqlalchemy import String, ForeignKey, Enum
from sqlalchemy.ext.associationproxy import association_proxy, AssociationProxy
from sqlalchemy.ext.hybrid import hybrid_property, hybrid_method
from sqlalchemy.orm import Mapped, mapped_column, relationship

from graphql_db.orm_base import ModelBase, UUIDType


# Model for hybrid property/method tests
class MixinTestPerson(ModelBase):
    __tablename__ = 'mixin_test_person'

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

    @hybrid_method
    def greeting(self, prefix: str) -> str:
        return f"{prefix} {self.first_name}"


# Models for AssociationProxy test
class MixinTestArticleTag(ModelBase):
    __tablename__ = 'mixin_test_article_tag'

    article_id: Mapped[str] = mapped_column(
        UUIDType, ForeignKey('mixin_test_article.id'), nullable=False
    )
    tag_name: Mapped[str] = mapped_column(String(50))

    def __init__(self, tag_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.tag_name = tag_name


class MixinTestArticle(ModelBase):
    __tablename__ = 'mixin_test_article'

    title: Mapped[str] = mapped_column(String(100))
    article_tags: Mapped[List["MixinTestArticleTag"]] = relationship(
        "MixinTestArticleTag"
    )
    tags: AssociationProxy[List[str]] = association_proxy(
        "article_tags", "tag_name"
    )

    def __init__(self, title: str = "", **kwargs):
        super().__init__(**kwargs)
        self.title = title


# Model for Enum test
class MixinTestStatus(enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class MixinTestPost(ModelBase):
    __tablename__ = 'mixin_test_post'

    title: Mapped[str] = mapped_column(String(100))
    status: Mapped[MixinTestStatus] = mapped_column(
        Enum(MixinTestStatus), default=MixinTestStatus.DRAFT
    )

    def __init__(
        self,
        title: str = "",
        status: MixinTestStatus = MixinTestStatus.DRAFT,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.title = title
        self.status = status


# Model for graphql_exclude_fields test
class MixinTestSecret(ModelBase):
    __tablename__ = 'mixin_test_secret'

    public_field: Mapped[str] = mapped_column(String(50))
    secret_field: Mapped[str] = mapped_column(String(50))
    _private_field: Mapped[str] = mapped_column(String(50))

    def __init__(
        self,
        public_field: str = "",
        secret_field: str = "",
        _private_field: str = "",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.public_field = public_field
        self.secret_field = secret_field
        self._private_field = _private_field

    @classmethod
    def graphql_exclude_fields(cls) -> list[str]:
        return ["secret_field"]


class TestMixinFeatures:

    def test_hybrid_property(self):
        """hybrid_property should be exposed and work correctly"""

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> MixinTestPerson:
                return MixinTestPerson(first_name="John", last_name="Doe")

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

    def test_hybrid_method_excluded(self):
        """hybrid_method should NOT be exposed on schema (requires arguments)"""

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def person(self) -> MixinTestPerson:
                return MixinTestPerson(first_name="John", last_name="Doe")

        # Query the hybrid_property (should work)
        gql_query_property = '''
            query GetPerson {
                person {
                    firstName
                    fullName
                }
            }
        '''

        result = schema.executor().execute(gql_query_property)
        assert result.errors is None or len(result.errors) == 0
        assert result.data["person"]["fullName"] == "John Doe"

        # Query the hybrid_method (should fail - not in schema)
        gql_query_method = '''
            query GetPerson {
                person {
                    greeting
                }
            }
        '''

        result_method = schema.executor().execute(gql_query_method)

        # hybrid_method should NOT be in schema
        assert result_method.errors is not None and len(result_method.errors) > 0

    def test_association_proxy(self):
        """association_proxy should be exposed as a list field"""

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def article(self) -> MixinTestArticle:
                article = MixinTestArticle(title="Test Article")
                return article

        gql_query = '''
            query GetArticle {
                article {
                    title
                    tags
                }
            }
        '''

        result = schema.executor().execute(gql_query)

        assert result.errors is None or len(result.errors) == 0
        assert result.data["article"]["title"] == "Test Article"

    def test_enum_column(self):
        """Enum columns should be exposed and return the enum name"""

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def post(self) -> MixinTestPost:
                return MixinTestPost(
                    title="My Post",
                    status=MixinTestStatus.PUBLISHED
                )

        gql_query = '''
            query GetPost {
                post {
                    title
                    status
                }
            }
        '''

        result = schema.executor().execute(gql_query)

        assert result.errors is None or len(result.errors) == 0
        assert result.data["post"]["title"] == "My Post"

    def test_graphql_exclude_fields(self):
        """graphql_exclude_fields should exclude specified fields from schema"""

        schema = GraphQLAPI()

        @schema.type(is_root_type=True)
        class Root:

            @schema.field
            def secret(self) -> MixinTestSecret:
                return MixinTestSecret(
                    public_field="visible",
                    secret_field="hidden",
                    _private_field="also_hidden"
                )

        # Query only public_field - secret_field should be excluded
        gql_query = '''
            query GetSecret {
                secret {
                    publicField
                }
            }
        '''

        result = schema.executor().execute(gql_query)

        assert result.errors is None or len(result.errors) == 0
        assert result.data["secret"]["publicField"] == "visible"

        # Verify secret_field is NOT in the schema
        gql_query_secret = '''
            query GetSecret {
                secret {
                    secretField
                }
            }
        '''

        result_secret = schema.executor().execute(gql_query_secret)

        # Should have an error because secretField doesn't exist
        assert result_secret.errors is not None and len(result_secret.errors) > 0
