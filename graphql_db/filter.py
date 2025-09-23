"""Basic filter implementation to replace sqlalchemy_orm.filter"""
from sqlalchemy import and_
from sqlalchemy.orm import Query


class Filter:
    """Simple filter class for SQLAlchemy queries."""

    def __init__(self, conditions=None):
        self.conditions = conditions or []

    def apply(self, query: Query) -> Query:
        """Apply filter conditions to a query."""
        if self.conditions:
            return query.filter(and_(*self.conditions))
        return query

    def add_condition(self, condition):
        """Add a condition to the filter."""
        self.conditions.append(condition)
