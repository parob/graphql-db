from context_helper import Context
from werkzeug import Request, Response

from graphql_db import DatabaseManager


class DatabaseContextMiddleware:

    def __init__(
            self,
            app,
            database_manager: DatabaseManager,
            eralchemy_path: str = None,
            context_key: str = "db_session"
    ):
        self.app = app
        self.database_manager = database_manager
        self.eralchemy_path = eralchemy_path
        self.context_key = context_key

    def __call__(
            self,
            environ,
            start_response
    ):
        request = Request(environ)

        if request.path == f"{self.eralchemy_path}":
            # Expose the service management HTTP server
            return Response(
                self.database_manager.db.entity_relationship_diagram(),
                content_type="text/html"
            )

        db_session = self.database_manager.db.session()

        try:
            with Context(**{self.context_key: db_session}):
                response = self.app(environ, start_response)

        except Exception as err:
            db_session.rollback()
            raise err
        else:
            db_session.commit()
        finally:
            db_session.close()

        return response
