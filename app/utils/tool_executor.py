import asyncio
import logging
from typing import Any

from ..models.orchestration_sgr import Parameter, ToolCallRequest
from ..models.tool_models import ToolResult
from ..models.ws_models import StatusCallback, StatusNotifier
from ..tools.tool_factory import ToolFactory
from .status_messages import get_tool_detail


logger = logging.getLogger(__name__)


def convert_parameters_to_dict(parameters: list[Parameter]) -> dict[str, Any]:
    return {param.name: param.value for param in parameters}


async def execute_tool_call(
    tool_call: ToolCallRequest,
    tool_factory: ToolFactory | None = None,
    on_status: StatusCallback = None,
) -> ToolResult:
    logger.info(
        f"tool_executor_001: Executing tool: \033[36m{tool_call.tool_name}\033[0m"
    )
    if tool_factory is None:
        tool_factory = ToolFactory()
    notifier = StatusNotifier(on_status)
    arguments_dict = convert_parameters_to_dict(tool_call.arguments)
    detail = get_tool_detail(tool_call.tool_name, arguments_dict)
    await notifier.emit(
        "tools",
        "started",
        detail or f"Calling {tool_call.tool_name}",
        detail=detail,
    )
    logger.info(f"tool_executor_002: Arguments: \033[33m{arguments_dict}\033[0m")
    try:
        result = await tool_factory.execute_tool(
            tool_name=tool_call.tool_name,
            tool_arguments=arguments_dict,
        )
        logger.info(
            f"tool_executor_003: Tool \033[36m{tool_call.tool_name}\033[0m executed successfully"
        )
        await notifier.emit(
            "tools",
            "completed",
            detail or f"{tool_call.tool_name} completed",
            detail=detail,
        )
        return ToolResult(
            tool_name=tool_call.tool_name,
            success=True,
            output=result,
        )
    except Exception as e:
        logger.error(
            f"tool_executor_error_001: Tool \033[31m{tool_call.tool_name}\033[0m failed: {e}"
        )
        await notifier.emit(
            "tools",
            "failed",
            f"{tool_call.tool_name} failed: {e!s}",
            detail=detail,
        )
        return ToolResult(
            tool_name=tool_call.tool_name,
            success=False,
            output={},
            error=str(e),
        )


async def execute_tool_calls(
    tool_calls: list[ToolCallRequest],
    tool_factory: ToolFactory | None = None,
    on_status: StatusCallback = None,
) -> list[ToolResult]:
    logger.info(f"tool_executor_004: Executing \033[33m{len(tool_calls)}\033[0m tools")
    if tool_factory is None:
        tool_factory = ToolFactory()
    tasks = [
        execute_tool_call(tool_call, tool_factory, on_status)
        for tool_call in tool_calls
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successful = sum(1 for r in results if isinstance(r, ToolResult) and r.success)
    logger.info(
        f"tool_executor_005: Completed: \033[32m{successful}\033[0m successful, \033[31m{len(results) - successful}\033[0m failed"
    )
    return [
        (
            r
            if isinstance(r, ToolResult)
            else ToolResult(
                tool_name="unknown",
                success=False,
                output={},
                error=str(r),
            )
        )
        for r in results
    ]
