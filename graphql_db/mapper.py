import typing
from inspect import isclass
from typing import Union

from graphql import GraphQLString
from graphql.type.definition import GraphQLList, GraphQLScalarType, GraphQLType
from graphql_api.mapper import GraphQLTypeMapper
from graphql_api.types import GraphQLUUID
from sqlalchemy import UUID, Column, Enum
from sqlalchemy.ext.associationproxy import AssociationProxy
from sqlalchemy.orm import RelationshipProperty, interfaces, relationship
from sqlalchemy.sql.type_api import TypeEngine


class HybridWrapper:

    def __init__(self, prop):
        self.property = prop


class AssociationWrapper:

    def __init__(self, prop, key, parent):
        self.property = prop
        self.key = key
        self.parent = parent


prop_types = Union[
    Column,
    RelationshipProperty,
    HybridWrapper,
    AssociationWrapper
]


class GraphQLSQLAlchemyHelpers:

    @staticmethod
    def map(
        prop: prop_types,
        mapper: GraphQLTypeMapper
    ) -> GraphQLType:
        if isinstance(prop, Column):
            return GraphQLSQLAlchemyHelpers.map_column(prop.type, mapper)

        elif isinstance(prop, RelationshipProperty):
            return GraphQLSQLAlchemyHelpers.map_relationship(prop, mapper)

        elif isinstance(prop, HybridWrapper):
            return GraphQLSQLAlchemyHelpers.map_hybrid(prop, mapper)

        elif isinstance(prop, AssociationWrapper):
            return GraphQLSQLAlchemyHelpers.map_association(prop, mapper)

        return GraphQLString

    @staticmethod
    def map_column(
        column_type: Column,
        mapper: GraphQLTypeMapper
    ) -> GraphQLType:
        try:
            python_type = column_type.python_type
        except NotImplementedError:
            try:
                python_type = column_type.impl.python_type
            except NotImplementedError:
                if isclass(column_type):
                    column_type_class = column_type
                else:
                    column_type_class = type(column_type)

                if issubclass(column_type_class, TypeEngine):
                    return GraphQLSQLAlchemyHelpers.map_type(column_type_class)

        return mapper.map(python_type)

    @staticmethod
    def map_type(type_: type[TypeEngine]) -> GraphQLScalarType:
        scalar_map = [
            ([UUID], GraphQLUUID),
            ([Enum], GraphQLString)
        ]

        for test_types, graphql_type in scalar_map:
            for test_type in test_types:
                if issubclass(type_, test_type):
                    return graphql_type

    @staticmethod
    def map_hybrid(
        hybrid_type: HybridWrapper,
        mapper: GraphQLTypeMapper
    ) -> GraphQLScalarType:

        prop = hybrid_type.property
        # hybrid_property stores the actual function in fget
        func = getattr(prop, 'fget', prop)
        type_hints = typing.get_type_hints(func)
        scalar_type: GraphQLScalarType = mapper.map(
            type_hints.pop('return', None)
        )
        return scalar_type

    @staticmethod
    def map_association(
        association_type: AssociationWrapper,
        mapper: GraphQLTypeMapper
    ) -> GraphQLType:

        association_proxy: AssociationProxy = association_type.property
        target_relationship: relationship = getattr(
            association_type.parent,
            association_proxy.target_collection
        )
        target_class = target_relationship.mapper.entity

        association_target_property = getattr(
            target_class,
            association_proxy.value_attr
        )

        column = association_target_property.property.columns[0]
        scalar_type: GraphQLScalarType = GraphQLSQLAlchemyHelpers.map(
            column,
            mapper
        )

        return GraphQLList(scalar_type)

    @staticmethod
    def map_relationship(
        relationship: RelationshipProperty,
        mapper: GraphQLTypeMapper
    ) -> GraphQLType:

        direction = relationship.direction
        model = relationship.mapper.entity

        graphql_type = mapper.map(model)
        if graphql_type:
            if not direction:
                return GraphQLList(graphql_type)
            if direction == interfaces.MANYTOONE or not relationship.uselist:
                return graphql_type
            elif direction in (interfaces.ONETOMANY, interfaces.MANYTOMANY):
                return GraphQLList(graphql_type)
