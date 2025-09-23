"""Basic order by implementation to replace sqlalchemy_orm.order_by"""
from enum import Enum
from typing import Any

from sqlalchemy.orm import Query


class OrderByDirection(Enum):
    """Direction for ordering."""
    asc = "ASC"
    desc = "DESC"


class OrderBy:
    """Simple order by class for SQLAlchemy queries."""

    def __init__(
        self, key: str, direction: OrderByDirection = OrderByDirection.asc
    ):
        self.key = key
        self.direction = direction

    def apply(self, query: Query, models: list[Any]) -> Query:
        """Apply ordering to a query."""
        if not models:
            return query

        model = models[0]
        field = getattr(model, self.key)

        if self.direction == OrderByDirection.asc:
            return query.order_by(field.asc())
        else:
            return query.order_by(field.desc())
