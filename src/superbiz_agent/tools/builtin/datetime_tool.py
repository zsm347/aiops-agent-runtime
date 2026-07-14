from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from superbiz_agent.tools.policies import DangerLevel, ToolPolicy
from superbiz_agent.tools.registry import ToolDefinition


class GetCurrentDateTimeArgs(BaseModel):
    timezone: str = "Asia/Shanghai"


async def get_current_datetime(args: GetCurrentDateTimeArgs) -> dict:
    try:
        tzinfo = ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError:
        tzinfo = ZoneInfo("Asia/Shanghai")

    now = datetime.now(tzinfo)
    return {
        "success": True,
        "data": {
            "timezone": args.timezone,
            "isoTime": now.isoformat(),
            "epochMillis": int(now.timestamp() * 1000),
        },
    }


def build_datetime_tool() -> ToolDefinition:
    return ToolDefinition(
        name="getCurrentDateTime",
        description="Return the current date and time for a requested IANA timezone.",
        args_model=GetCurrentDateTimeArgs,
        handler=get_current_datetime,
        policy=ToolPolicy(
            tool_name="getCurrentDateTime",
            danger_level=DangerLevel.LOW,
            timeout_seconds=1,
            max_retries=0,
            idempotent=True,
        ),
    )
