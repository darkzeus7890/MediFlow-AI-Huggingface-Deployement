# app.py
import uuid
import asyncio
import traceback
import gradio as gr

from agent import AGENTS, make_runner_for, session_service, APP_NAME, USER_ID
from google.genai.types import Content, Part

SESSION_ID_PREFIX = "session-"  # prefix used in hidden marker
DEFAULT_AGENT_KEY = "triage"

# Helper: ensure session creation for server-side services
def ensure_session_for(agent_key=DEFAULT_AGENT_KEY, session_id="global-startup"):
    runner = make_runner_for(AGENTS[agent_key])
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id)
        )
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()

# create a lightweight global startup session to satisfy services (optional)
ensure_session_for()

# Async helper: sends content via the runner and returns final response
async def _send_and_get_final_response_async(message: str, agent_key: str, session_id: str):
    runner = make_runner_for(AGENTS[agent_key])
    user_input = Content(parts=[Part(text=message)], role="user")
    final_response = "(no response)"
    try:
        async for event in runner.run_async(user_id=USER_ID, session_id=session_id, new_message=user_input):
            if event.is_final_response() and getattr(event, "content", None) and getattr(event.content, "parts", None):
                final_response = event.content.parts[0].text
    except Exception as exc:
        final_response = f"[Agent runtime error] {exc}\n{traceback.format_exc()}"
    return final_response

# Sync wrapper to call async helper safely from Gradio
def run_agent_sync(message: str, agent_key: str, session_id: str):
    try:
        return asyncio.run(_send_and_get_final_response_async(message, agent_key, session_id))
    except RuntimeError:
        # if event loop is already running, create a separate loop for this call
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_send_and_get_final_response_async(message, agent_key, session_id))
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
    except Exception as exc:
        return f"[Error invoking agent] {exc}\n{traceback.format_exc()}"

# Chat callback used by Gradio ChatInterface
def chat_fn(message, history, agent_choice=DEFAULT_AGENT_KEY):
    """
    message: user string
    history: list of tuples (user, bot) OR may include a hidden session marker as first element
    agent_choice: which agent key to run
    Returns: response string (ChatInterface will append to chat UI)
    """

    # 1) Detect or create per-visitor session id.
    # If the very first item in history is our hidden marker, use it.
    # Hidden marker format (internal): ("__SID__:<session-id>", "") as the first tuple.
    session_id = None
    try:
        if history and isinstance(history, list) and len(history) > 0:
            first = history[0]
            if isinstance(first, (list, tuple)) and len(first) >= 1 and isinstance(first[0], str):
                if first[0].startswith("__SID__:"):
                    session_id = first[0].split(":", 1)[1]  # extract id
                    # remove the hidden marker from history so it is not rendered
                    history = history[1:]
    except Exception:
        # if anything odd happens, fallback to new session id
        session_id = None

    if not session_id:
        # generate new session id for this visitor
        session_id = SESSION_ID_PREFIX + str(uuid.uuid4())

        # create the session on the runner (best-effort; ignore if already exists)
        try:
            runner = make_runner_for(AGENTS[agent_choice])
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(runner.session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=session_id))
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                loop.close()
        except Exception:
            # ignore errors here; the call will still attempt to run
            pass

        # Prepend hidden session marker into the history that will be sent back to client.
        # We send it back so subsequent calls from the same browser include the marker.
        # Marker looks like ("__SID__:<id>", "") and will be stripped on receipt by this function.
        # ChatInterface will not display a blank assistant string, so marker remains hidden.
        if history is None:
            history_to_return = [ (f"__SID__:{session_id}", "") ]
        else:
            history_to_return = [ (f"__SID__:{session_id}", ""), *history ]
    else:
        # keep existing history as-is
        history_to_return = history

    # 2) Run the agent synchronously (under the hood we call the async runner)
    response = run_agent_sync(message, agent_choice, session_id)

    # 3) Return response (ChatInterface appends it to the chat)
    # Also return the history with the hidden marker so the browser keeps the session id across refreshes.
    # ChatInterface expects a single return value (string). However we can return the response only.
    # To ensure the hidden marker survives, we rely on ChatInterface preserving message history on client.
    # Therefore we return only the response string here.
    return response

# Build Gradio UI (minimal; Gradio v5)
def main():
    with gr.Blocks() as demo:
        gr.Markdown("### 🤖 MediFlow — ADK (per-visitor session)")
        gr.ChatInterface(fn=chat_fn, title="MediFlow Agent")
    demo.launch()

if __name__ == "__main__":
    main()
