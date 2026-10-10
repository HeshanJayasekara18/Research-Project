"""One JSON error envelope for every failure: {"error": {code, message, request_id, details}}."""
from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = logging.getLogger('service')


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def request_id(request: Request) -> str:
    rid = getattr(request.state, 'request_id', None)
    return rid or uuid.uuid4().hex


def _envelope(request: Request, status: int, code: str, message: str, details=None) -> JSONResponse:
    rid = request_id(request)
    body = {'error': {'code': code, 'message': message, 'request_id': rid}}
    if details is not None:
        body['error']['details'] = details
    return JSONResponse(body, status_code=status, headers={'X-Request-ID': rid})


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError):
        return _envelope(request, exc.status, exc.code, exc.message, exc.details)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = [{'loc': list(e.get('loc', ())), 'msg': e.get('msg', ''), 'type': e.get('type', '')}
                   for e in exc.errors()]
        return _envelope(request, 422, 'invalid_request', 'Request body failed validation.', details)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: 'not_found', 405: 'method_not_allowed'}.get(exc.status_code, 'http_error')
        return _envelope(request, exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception('Unhandled error (request_id=%s)', request_id(request))
        return _envelope(request, 500, 'internal_error', 'Internal server error.')
