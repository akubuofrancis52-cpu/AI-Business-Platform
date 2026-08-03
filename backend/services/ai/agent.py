import json
import re

from services.ai.openai_provider import OpenAIProvider
from services.ai.order_executor import confirm_pending_order

from services.ai.agent_tools import (
    tool_search_menu,
    get_customer,
    get_active_order,
    get_pending_order,
    create_order_preview,
    discard_pending_order,
    modify_active_order,
    cancel_active_order
)


# ==========================
# JSON & XML PARSERS
# ==========================

def extract_json_objects(text):

    if not text:
        return []

    text = text.strip()

    text = text.replace(
        "```json",
        ""
    )

    text = text.replace(
        "```",
        ""
    )

    text = text.strip()

    try:

        parsed = json.loads(text)

        if isinstance(parsed, list):

            return [
                item
                for item in parsed
                if isinstance(item, dict)
            ]

        if isinstance(parsed, dict):

            return [parsed]

    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    objects = []
    position = 0

    while position < len(text):

        start = text.find(
            "{",
            position
        )

        if start == -1:
            break

        try:

            obj, end = decoder.raw_decode(
                text[start:]
            )

            if isinstance(obj, dict):

                objects.append(obj)

            position = start + end

        except json.JSONDecodeError:

            position = start + 1

    return objects


def extract_xml_tool_calls(text):
    """
    Parses LLMs that output tool calls using XML tags instead of JSON.
    """
    if not text:
        return []

    commands = []
    pattern = r"<tool_call>\s*([a-zA-Z0-9_]+)(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    for tool_name, inner in matches:
        args = {}
        arg_pattern = r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>"
        arg_matches = re.findall(arg_pattern, inner, re.DOTALL)
        
        for k, v in arg_matches:
            args[k.strip()] = v.strip()

        commands.append({
            "type": "tool_call",
            "tool": tool_name.strip(),
            "arguments": args
        })

    return commands


# ==========================
# CONFIRMATION DETECTION
# ==========================

def is_confirmation(message):

    text = message.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    confirmations = {
        "yes",
        "yes please",
        "yes pls",
        "confirm",
        "confirmed",
        "confirm it",
        "go ahead",
        "go for it",
        "place it",
        "place the order",
        "do it",
        "that's correct",
        "that is correct",
        "correct",
        "okay",
        "ok",
        "sure",
        "sure thing",
        "sounds good"
    }

    return text in confirmations


# ==========================
# REJECTION DETECTION
# ==========================

def is_rejection(message):

    text = message.lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    rejections = {
        "no",
        "no thanks",
        "no thank you",
        "cancel",
        "never mind",
        "nevermind",
        "forget it",
        "don't place it",
        "do not place it"
    }

    return text in rejections


# ==========================
# UTILS
# ==========================

def _extract_pending_items(pending_data):
    if not isinstance(pending_data, dict):
        return []
    if "items" in pending_data:
        return pending_data["items"]
    if "order" in pending_data and isinstance(pending_data["order"], dict):
        return pending_data["order"].get("items", [])
    if "preview" in pending_data and isinstance(pending_data["preview"], dict):
        return pending_data["preview"].get("items", [])
    return []


# ==========================
# TOOL EXECUTION
# ==========================

def execute_tool(
    business_id,
    phone,
    message,
    tool,
    arguments
):

    if tool == "search_menu":

        query = arguments.get("query") or arguments.get("message") or message

        return tool_search_menu(
            business_id,
            query
        )

    if tool == "get_customer":

        return get_customer(
            business_id,
            phone
        )

    if tool == "get_active_order":

        return get_active_order(
            business_id,
            phone
        )

    if tool == "get_pending_order":

        return get_pending_order(
            business_id,
            phone
        )

    if tool == "create_order_preview":

        order_message = arguments.get("message") or arguments.get("query") or message

        return create_order_preview(
            business_id,
            phone,
            order_message
        )

    if tool == "confirm_order":
        
        pending = get_pending_order(business_id, phone)
        items = _extract_pending_items(pending)

        return confirm_pending_order(
            business_id,
            phone,
            items
        )

    if tool == "discard_order_preview":

        return discard_pending_order(
            business_id,
            phone
        )

    if tool == "modify_order":

        mod_message = arguments.get("message") or arguments.get("query") or message

        return modify_active_order(
            business_id,
            phone,
            mod_message
        )

    if tool == "cancel_order":

        return cancel_active_order(
            business_id,
            phone
        )

    return {
        "success": False,
        "message": (
            "That tool is not available."
        )
    }


# ==========================
# FORMAT MEMORY
# ==========================

def format_history(
    history
):

    if not history:
        return "No previous conversation."

    parts = []

    for item in history:

        parts.append(
            f"""
Customer:
{item.get("message", "")}

Assistant:
{item.get("response", "")}
"""
        )

    return "\n--------------------\n".join(
        parts
    )


# ==========================
# AGENT
# ==========================

def run_agent(
    business_id,
    phone,
    message,
    language="English",
    history=None
):

    provider = OpenAIProvider()

    history = history or []

    history_text = format_history(
        history
    )

    # ==========================
    # CHECK PENDING PREVIEW
    # ==========================

    pending = get_pending_order(
        business_id,
        phone
    )

    # ==========================
    # DETERMINISTIC CONFIRMATION
    # ==========================

    if pending.get("success"):

        if is_confirmation(message):
            
            items = _extract_pending_items(pending)
            
            result = confirm_pending_order(
                business_id,
                phone,
                items
            )

            return build_backend_response(
                provider,
                message,
                language,
                "confirm_order",
                result,
                history_text
            )

        if is_rejection(message):

            result = discard_pending_order(
                business_id,
                phone
            )

            return build_backend_response(
                provider,
                message,
                language,
                "discard_order_preview",
                result,
                history_text
            )

    pending_context = ""

    if pending.get("success"):

        pending_context = f"""

PENDING ORDER PREVIEW

{json.dumps(
    pending,
    ensure_ascii=False,
    indent=2
)}

This preview is waiting for confirmation.

"""

    # ==========================
    # MAIN AGENT PROMPT
    # ==========================

    system_prompt = f"""
You are the central AI agent for a restaurant.

CUSTOMER
Phone:
{phone}

Language:
{language}

CURRENT MESSAGE:
{message}

RECENT CONVERSATION:
{history_text}

{pending_context}

YOUR RESPONSIBILITIES

You can:
- answer restaurant questions
- search the menu using search_menu
- create order previews using create_order_preview when the customer orders food
- confirm pending orders
- reject pending orders
- modify active orders
- cancel active orders

IMPORTANT ORDER RULES

1. When the customer says things like "I want...", "I'll take...", "give me...", or lists items like "2 shawarma 1 ice cream", you MUST call the **create_order_preview** tool. Do NOT just search the menu.
2. Do NOT create a real Order from an initial request. Real order creation requires confirmation.
3. A pending preview must be respected.
4. Never invent menu items or prices.

AVAILABLE TOOLS

search_menu
get_customer
get_active_order
get_pending_order
create_order_preview
confirm_order
discard_order_preview
modify_order
cancel_order

OUTPUT FORMATS

You can output tool calls using JSON or XML tags.

EXAMPLE 1 (Creating an order preview - JSON):
{{
    "type": "tool_call",
    "tool": "create_order_preview",
    "arguments": {{
        "message": "{message}"
    }}
}}

EXAMPLE 2 (Creating an order preview - XML):
<tool_call>create_order_preview
<arg_key>message</arg_key>
<arg_value>{message}</arg_value>
</tool_call>

For a normal answer:
{{
    "type": "response",
    "message": "..."
}}

Return ONLY JSON or valid tool call format.
"""

    first_response = provider.generate(
        system_prompt
    )

    commands = extract_json_objects(first_response)
    if not commands:
        commands = extract_xml_tool_calls(first_response)

    # ==========================
    # INVALID MODEL OUTPUT
    # ==========================

    if not commands:

        cleaned = first_response.strip()

        if cleaned:

            return {
                "type": "response",
                "message": cleaned
            }

        return {
            "type": "response",
            "message": (
                "I couldn't process that request."
            )
        }

    # ==========================
    # FRONTEND ACTION
    # ==========================

    for command in commands:

        if command.get(
            "type"
        ) == "frontend_action":

            allowed = {
                "open_dashboard",
                "open_orders",
                "open_customers",
                "open_menu",
                "open_ai",
                "open_settings"
            }

            action = command.get(
                "action"
            )

            if action in allowed:

                return {
                    "type": "frontend_action",
                    "action": action,
                    "message": (
                        "Opening that section."
                    )
                }

    # ==========================
    # TOOL CALLS
    # ==========================

    tool_results = []

    for command in commands:

        if command.get(
            "type"
        ) != "tool_call":

            continue

        tool = command.get(
            "tool"
        )

        arguments = command.get(
            "arguments",
            {}
        )

        if not isinstance(
            arguments,
            dict
        ):
            arguments = {}

        allowed_tools = {
            "search_menu",
            "get_customer",
            "get_active_order",
            "get_pending_order",
            "create_order_preview",
            "confirm_order",
            "discard_order_preview",
            "modify_order",
            "cancel_order"
        }

        if tool not in allowed_tools:
            continue

        result = execute_tool(
            business_id,
            phone,
            message,
            tool,
            arguments
        )

        tool_results.append({
            "tool": tool,
            "result": result
        })

    # ==========================
    # NO TOOL USED
    # ==========================

    if not tool_results:

        for command in commands:

            if command.get(
                "type"
            ) == "response":

                return {
                    "type": "response",
                    "message": command.get(
                        "message",
                        "I couldn't help with that."
                    )
                }

        return {
            "type": "response",
            "message": (
                "I couldn't understand "
                "that request."
            )
        }

    # ==========================
    # FINAL RESPONSE
    # ==========================

    return build_backend_response(
        provider,
        message,
        language,
        "multiple_tools",
        tool_results,
        history_text
    )


# ==========================
# FINAL RESPONSE GENERATOR
# ==========================

def build_backend_response(
    provider,
    message,
    language,
    tool,
    result,
    history_text
):

    final_prompt = f"""
You are the final customer-facing restaurant assistant.

CUSTOMER MESSAGE:

{message}

RECENT CONVERSATION:

{history_text}

BACKEND RESULT:

{json.dumps(
    result,
    ensure_ascii=False,
    indent=2
)}

RULES:

- Never show JSON.
- Never mention tools.
- Never mention backend processing.
- Never invent data.
- Use only information in the backend result
  and recent conversation.
- Preserve context from the conversation.
- If a preview was created, clearly ask for confirmation.
- If an order was confirmed, provide the real order ID
  and total.
- If an order was rejected, say so clearly.
- Keep the response natural and concise.
- Reply in {language}.

Return ONLY:

{{
    "type": "response",
    "message": "..."
}}
"""

    final_response = provider.generate(
        final_prompt
    )

    commands = extract_json_objects(
        final_response
    )

    for command in commands:

        if command.get(
            "type"
        ) == "response":

            message_text = command.get(
                "message"
            )

            if message_text:

                return {
                    "type": "response",
                    "message": str(
                        message_text
                    )
                }

    cleaned = final_response.strip()

    return {
        "type": "response",
        "message": cleaned
    }