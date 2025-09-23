import base64
import json
from typing import Optional
from uuid import UUID

from graphql_api import GraphQLAPI, GraphQLError
from sqlalchemy.exc import NoResultFound

from graphql_db.mixin import GraphQLSQLAlchemyMixin

try:
    from sqlalchemy import and_, or_

except ImportError:
    raise ImportError("sqlalchemy package not found")

from graphql_api.relay import Connection, Edge, Node, PageInfo
from graphql_api.utils import to_snake_case

from graphql_db.filter import Filter
from graphql_db.order_by import OrderBy, OrderByDirection


@GraphQLAPI.type(abstract=True)
class RelayBase(Node, GraphQLSQLAlchemyMixin):
    id: UUID

    @classmethod
    def graphql_from_input(cls, id: UUID):
        value = cls.get(id)

        if value is None:
            raise NoResultFound(
                f"{cls.__name__} with UUID {id} was not found."
            )

    @classmethod
    def get(cls, id: UUID = None) -> Optional['Node']:
        if id:
            return cls.filter(id=id).one_or_none()


def relay_connection(relay_model: Node = None):

    if relay_model:

        class ModelEdge(Edge):

            @GraphQLAPI.field
            def node(self) -> relay_model:
                return self.node

        ModelEdge.__name__ = relay_model.__name__ + "Edge"

    else:
        ModelEdge = Edge

    class SQLConnection(Connection):
        """
        The `SQLConnection` Object type represents a Relay Connection.
        `SQLConnection` contains `OrderBy` and `Filter` attributes
        to order and filter the Edges.
        `https://facebook.github.io/relay/graphql/
        connections.htm#sec-Connection-Types`
        """

        @staticmethod
        def encode_cursor(instance, order_by: list[OrderBy]):
            cursor = {"type": "order_by", "criterion": []}

            for order_by_item in order_by:
                order_by_key = to_snake_case(order_by_item.key)
                instance_value = getattr(instance, order_by_key)

                if order_by_item.direction == OrderByDirection.asc:
                    operator = "gt"
                else:
                    operator = "lt"

                cursor['criterion'].append({
                    "value": str(instance_value),
                    "field": order_by_item.key,
                    "operator": operator
                })

            json_cursor = json.dumps(cursor)
            base64_json_cursor = base64.b64encode(json_cursor.encode('utf-8'))
            return base64_json_cursor.decode('utf-8')

        @staticmethod
        def decode_cursor(base64_json_cursor: str) -> dict:
            try:
                json_cursor_bytes = base64_json_cursor.encode('utf-8')
                json_cursor = base64.decodebytes(json_cursor_bytes)
                cursor = json.loads(json_cursor)
                return cursor
            except UnicodeDecodeError as err:
                raise GraphQLError(
                    f"Cursor {base64_json_cursor} was an "
                    f"invalid cursor encoding, {err}."
                )

        def apply_cursor(self, query, cursor, flip=False):
            cursor = SQLConnection.decode_cursor(cursor)

            if cursor.get('type') == 'order_by':
                criterion = cursor.get('criterion')

                or_expressions = []
                and_expressions = []

                for clause in criterion:
                    value = clause['value']
                    field = clause['field']
                    operator = clause['operator']

                    if (operator == "gt" and not flip) or \
                            (operator == "lt" and flip):
                        or_expression = getattr(self.model, field) > value
                    else:
                        or_expression = getattr(self.model, field) < value

                    or_expressions.append(or_expression)

                    if and_expressions:
                        or_expressions.append(
                            and_(or_expression, *and_expressions)
                        )

                    and_expressions.append(getattr(self.model, field) == value)

                return query.filter(or_(*or_expressions))
            else:
                raise GraphQLError(
                    f"Cursor {cursor} was an invalid cursor format."
                )

        def __init__(
            self,
            model: type[Node] = relay_model,
            order_by: list[OrderBy] = None,
            filter: Filter = None,
            before: str = None,
            after: str = None,
            first: int = None,
            last: int = None
        ):
            super().__init__(
                before=before,
                after=after,
                first=first,
                last=last
            )

            self.model = model

            if not self.model:
                raise GraphQLError(
                    "The model must be specified for a connection.")

            if order_by is None:
                self.order_by = [OrderBy('id', OrderByDirection.asc)]
            else:
                self.order_by = order_by

            if first and last:
                raise GraphQLError(
                    "Including first and last is strongly discouraged."
                )

            _query = self.model.query()

            if filter:
                _query = filter.apply(_query)

            self.count = _query.count()

            if after:
                _query = self.apply_cursor(_query, after)

            if before:
                _query = self.apply_cursor(_query, before, True)

            for order_by_item in self.order_by:
                _query = order_by_item.apply(_query, [self.model])

            self.offset = 0
            self.limit = 0

            if first:
                self.limit = first
                self.offset = 0

            self.page_count = _query.count()

            if last:
                self.limit = last
                self.offset = max(0, self.page_count - last)

            if self.offset:
                _query = _query.offset(self.offset)

            if self.limit:
                _query = _query.limit(self.limit)

            self.query = _query

        @GraphQLAPI.field
        def edges(self) -> list[ModelEdge]:
            return [
                Edge(instance, self.encode_cursor(instance, self.order_by))
                for instance in self.query.all()
            ]

        @GraphQLAPI.field
        def page_info(self) -> PageInfo:
            edges = self.edges()

            start_cursor = None
            end_cursor = None

            if edges:
                start_cursor = edges[0].cursor
                end_cursor = edges[-1].cursor

            has_previous_page = self.offset > 0 and self.page_count > 0
            if self.limit:
                has_next_page = self.offset + self.limit < self.page_count
            else:
                has_next_page = False

            return PageInfo(
                has_previous_page,
                has_next_page,
                start_cursor,
                end_cursor,
                self.page_count  # Add the total count
            )

    if relay_model:
        SQLConnection.__name__ = relay_model.__name__ + "Connection"

    return SQLConnection


SQLConnection = relay_connection()
