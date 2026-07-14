import json
from collections.abc import AsyncIterator
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from superbiz_agent.api.schemas import (
    ApiResponse,
    ChatRequest,
    ChatResponse,
    ClearRequest,
    SessionInfoResponse,
    SseMessage,
)
from superbiz_agent.config import Settings, get_settings
from superbiz_agent.harness.context import AgentRequestContext
from superbiz_agent.harness.service import AgentHarnessService
from superbiz_agent.security.auth import AuthContextResolver, AuthError
from superbiz_agent.security.permissions import (
    CHAT_INVOKE,
    SESSION_CLEAR,
    SESSION_READ,
    PermissionDenied,
    require_permission,
)

router = APIRouter(prefix="/api", tags=["chat"])


def resolve_request_context(request: Request, session_id: Optional[str]) -> AgentRequestContext:
    principal = _auth_resolver(request).resolve(request)
    return principal.to_request_context(session_id=session_id)


def resolve_authorized_context(
    request: Request,
    session_id: Optional[str],
    permission: str,
) -> AgentRequestContext:
    try:
        context = resolve_request_context(request, session_id)
        require_permission(context, permission, settings=_settings(request))
        return context
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=exc.message) from exc
    except PermissionDenied as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def get_harness_service(request: Request) -> AgentHarnessService:
    service = getattr(request.app.state, "harness_service", None)
    if isinstance(service, AgentHarnessService):
        return service
    service = AgentHarnessService.placeholder()
    request.app.state.harness_service = service
    return service


def _settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    return settings if isinstance(settings, Settings) else get_settings()


def _auth_resolver(request: Request) -> AuthContextResolver:
    resolver = getattr(request.app.state, "auth_resolver", None)
    if isinstance(resolver, AuthContextResolver):
        return resolver
    resolver = AuthContextResolver(_settings(request))
    request.app.state.auth_resolver = resolver
    return resolver


@router.post("/chat", response_model=ApiResponse)
async def chat(payload: ChatRequest, request: Request) -> ApiResponse:
    context = resolve_authorized_context(request, payload.session_id, CHAT_INVOKE)
    if payload.question is None or not payload.question.strip():
        return ApiResponse.success(ChatResponse.error("问题内容不能为空").model_dump())

    service = get_harness_service(request)
    result = await service.chat(context, payload.question)
    if result.success:
        return ApiResponse.success(ChatResponse.ok(result.answer or "").model_dump())
    return ApiResponse.success(
        ChatResponse.error(result.error_message or "Agent Harness failed").model_dump()
    )


@router.post("/chat_stream")
async def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    context = resolve_authorized_context(request, payload.session_id, CHAT_INVOKE)
    service = get_harness_service(request)
    question = payload.question or ""

    async def generate() -> AsyncIterator[str]:
        async for message in service.chat_stream(context, question):
            yield _encode_sse_message(message)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _encode_sse_message(message: SseMessage) -> str:
    payload = json.dumps(
        message.model_dump(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: message\ndata: {payload}\n\n"


@router.post("/chat/clear", response_model=ApiResponse)
async def clear_chat_history(payload: ClearRequest, request: Request) -> ApiResponse:
    if payload.session_id is None or not payload.session_id:
        return ApiResponse.error("会话ID不能为空")

    context = resolve_authorized_context(request, payload.session_id, SESSION_CLEAR)
    service = get_harness_service(request)
    await service.clear(context)
    return ApiResponse.success("会话历史已清空")


@router.get("/chat/session/{session_id}", response_model=ApiResponse)
async def get_session_info(session_id: str, request: Request) -> ApiResponse:
    context = resolve_authorized_context(request, session_id, SESSION_READ)
    service = get_harness_service(request)
    info = await service.session_info(context)
    return ApiResponse.success(
        SessionInfoResponse(
            sessionId=info.session_id,
            messagePairCount=info.message_pair_count,
            createTime=info.create_time,
        ).model_dump()
    )
