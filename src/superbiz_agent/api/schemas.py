from typing import Any
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: Optional[str] = Field(
        default=None,
        alias="Id",
        validation_alias=AliasChoices("Id", "id", "ID"),
    )
    question: Optional[str] = Field(
        default=None,
        alias="Question",
        validation_alias=AliasChoices("Question", "question", "QUESTION"),
    )


class ClearRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: Optional[str] = Field(
        default=None,
        alias="Id",
        validation_alias=AliasChoices("Id", "id", "ID"),
    )


class ChatResponse(BaseModel):
    success: bool
    answer: Optional[str] = None
    errorMessage: Optional[str] = None

    @classmethod
    def ok(cls, answer: str) -> "ChatResponse":
        return cls(success=True, answer=answer)

    @classmethod
    def error(cls, message: str) -> "ChatResponse":
        return cls(success=False, errorMessage=message)


class ApiResponse(BaseModel):
    code: int
    message: str
    data: Any = None

    @classmethod
    def success(cls, data: Any) -> "ApiResponse":
        return cls(code=200, message="success", data=data)

    @classmethod
    def error(cls, message: str) -> "ApiResponse":
        return cls(code=500, message=message, data=None)


class SseMessage(BaseModel):
    type: str
    data: Any = None


class SessionInfoResponse(BaseModel):
    sessionId: str
    messagePairCount: int
    createTime: int
