
from graphql import (
    GraphQLField,
    GraphQLInputField,
    GraphQLInputType,
    GraphQLNonNull,
    GraphQLObjectType,
    GraphQLOutputType,
    GraphQLType,
)
from graphql_api.mapper import GraphQLTypeMapper, GraphQLTypeWrapper
from graphql_api.utils import to_camel_case
from sqlalchemy import Column
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.ext.associationproxy import AssociationProxy
from sqlalchemy.ext.hybrid import hybrid_method, hybrid_property
from sqlalchemy.orm import RelationshipProperty

from graphql_db.mapper import (
    AssociationWrapper,
    GraphQLSQLAlchemyHelpers,
    HybridWrapper,
)


class GraphQLSQLAlchemyMixin(GraphQLTypeWrapper):
    """
    `GraphQLSQLAlchemyMixin` subclasses will map
    the following into properties on a GraphQLObject

        - Columns
        - Relationships
        - Hybrid Properties
        - Hybrid Methods
        - Association Objects

    This mixin will exclude private fields
    (field names that begin with an underscore),
    and exclude any fields that are listed in 'graphql_exclude_fields'
    """

    @classmethod
    def graphql_merge(cls, cls_):
        for key, member in cls_.__dict__.items():
            if hasattr(member, 'graphql'):
                setattr(cls, key, member)

    @classmethod
    def graphql_exclude_fields(cls) -> list[str]:
        return []

    @classmethod
    def graphql_type(cls, mapper: GraphQLTypeMapper) -> GraphQLObjectType:
        # noinspection PyTypeChecker
        base_type: GraphQLObjectType = mapper.map(cls, use_graphql_type=False)

        # Remove any modifiers
        while hasattr(base_type, 'of_type'):
            # noinspection PyUnresolvedReferences
            base_type = base_type.of_type

        if mapper.as_input:
            return base_type

        properties_type = dict[
            str,
            Column | RelationshipProperty | HybridWrapper
        ]

        properties: properties_type = {}
        inspected_model = sqlalchemy_inspect(cls)

        for name, item in inspected_model.columns.items():
            properties[name] = item

        for _relationship in inspected_model.relationships:
            properties[_relationship.key] = _relationship

        for name, item in inspected_model.all_orm_descriptors.items():
            if isinstance(item, (hybrid_method, hybrid_property)):
                properties[name] = HybridWrapper(getattr(cls, name))

            if isinstance(item, AssociationProxy):
                # noinspection PyTypeChecker
                properties[name] = AssociationWrapper(
                    getattr(cls, name),
                    name,
                    cls
                )

        exclude_fields = cls.graphql_exclude_fields()

        properties = {
            name: prop
            for name, prop in properties.items()
            if not name.startswith("_") and name not in exclude_fields
        }

        def local_fields_callback():
            local_type = base_type
            local_properties = properties
            local_mapper = mapper

            # noinspection PyProtectedMember
            local_type_fields = local_type._fields

            def fields_callback():
                local_fields = {}

                if local_type_fields:
                    try:
                        local_fields = local_type_fields()
                    except AssertionError:
                        pass

                for prop_name, prop in local_properties.items():

                    def local_resolver():
                        local_prop_name = prop_name

                        def resolver(
                            self,
                            info=None,
                            context=None,
                            *args,
                            **kwargs
                        ):
                            return getattr(self, local_prop_name)
                        return resolver

                    type_: GraphQLType = GraphQLSQLAlchemyHelpers.map(
                        prop,
                        local_mapper
                    )

                    if local_mapper.as_input:
                        type_: GraphQLInputType
                        field = GraphQLInputField(type_=type_)
                    else:
                        type_: GraphQLOutputType
                        if hasattr(prop, 'nullable') and not prop.nullable:
                            type_ = GraphQLNonNull(type_=type_)

                        field = GraphQLField(
                            type_=type_,
                            resolve=local_resolver()
                        )

                    local_fields[to_camel_case(prop_name)] = field

                return local_fields

            return fields_callback

        base_type._fields = local_fields_callback()

        return base_type
