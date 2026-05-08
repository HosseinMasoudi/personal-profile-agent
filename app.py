from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

def push_message(text: str):
    bot_url = os.getenv("BOT_URL")
    chat_id = os.getenv("CHAT_ID")

    if not bot_url or not chat_id:
        print("BOT_URL or CHAT_ID is not set; skipping notification.")
        return

    payload = {
        "chat_id": chat_id,
        "text": text,
    }
    try:
        response = requests.post(bot_url, json=payload, timeout=20)
        response.raise_for_status()
        try:
            print(response.json())
        except ValueError:
            print(response.text)
    except requests.RequestException as exc:
        print(f"Notification failed: {exc}")

def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    push_message(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question: str):
    push_message(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

tool_registry = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}


class Me:

    def __init__(self):
        self.openai = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("BASE_URL")
        )

        self.mode_name = os.getenv("MODEL") or os.getenv("model")
        if not self.mode_name:
            raise ValueError("Missing MODEL environment variable. Set MODEL in your .env file.")

        self.name = "Hossein Masoudi"
        reader = PdfReader(str(BASE_DIR / "MyResume.pdf"))
        self.resume = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.resume += text

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            print(f"Tool called: {tool_name}", flush=True)
            tool = tool_registry.get(tool_name)
            if tool is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = tool(**arguments)
                except Exception as exc:
                    result = {"error": f"Tool {tool_name} failed: {exc}"}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def normalize_history(self, history):
        normalized = []
        for item in history or []:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_message, assistant_message = item
                normalized.append({"role": "user", "content": user_message})
                if assistant_message is not None:
                    normalized.append({"role": "assistant", "content": assistant_message})
        return normalized

    def system_prompt(self):
        return (
            f"You are acting as {self.name}.\n"
            f"You are answering questions on {self.name}'s website.\n"
            f"Focus on career topics: background, skills, experience, projects, and fit for roles.\n\n"

            f"Context you must use:\n"
            f"- LinkedIn profile summary and website resume content provided below.\n\n"

            f"Communication style:\n"
            f"- Be professional, warm, and engaging.\n"
            f"- Write as {self.name} (first-person where appropriate).\n"
            f"- Keep answers concise and helpful.\n\n"

            f"Tool usage rules (must follow):\n"
            f"1) If you do NOT know the answer from the provided context, call "
            f"`record_unknown_question` with the user question (even if trivial/unrelated).\n"
            f"2) If the user shows interest or begins discussing next steps, ask for their email and call "
            f"`record_user_details` with the email they provide.\n\n"

            f"After answering:\n"
            f"- If appropriate, encourage the user to get in touch via email.\n"
            f"- Do not invent facts. If information is missing, say so briefly and rely on the tool rule above.\n\n"

            f"## Resume\n{self.resume}\n"
        )


    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(self.normalize_history(history))
        messages.append({"role": "user", "content": message})
        done = False
        while not done:
            response = self.openai.chat.completions.create(model=self.mode_name, messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content or ""
    

if __name__ == "__main__":
    me = Me()

    def chat_fn(message, history):
        reply = me.chat(message, history)
        return reply

    gr.ChatInterface(fn=chat_fn).launch()
