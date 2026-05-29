import json
import re
import os
import gradio as gr
import smtplib
import logging
from pypdf import PdfReader
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)

def is_valid_email(email: str) -> bool:
    email = email.strip()
    pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    return re.match(pattern, email) is not None

def send_email_via_smtp(subject: str, body: str):
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    notify_email = os.getenv("NOTIFY_EMAIL")

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, notify_email]):
        raise ValueError("SMTP configuration is incomplete.")

    message = f"Subject: {subject}\nFrom: {smtp_user}\nTo: {notify_email}\n\n{body}"

    logger.info("Connecting to SMTP server %s:%s", smtp_host, smtp_port)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as s:
        logging.info("SMTP connection established, starting TLS and logging in")
        s.set_debuglevel(1)
        s.starttls()
        s.login(smtp_user, smtp_pass)
        s.sendmail(smtp_user, notify_email, message)
        logger.info("Email sent to %s", notify_email)


def email_user_interaction(
    user_email: str,
    user_name: str = "Name not provided",
    questions_asked: str = "Not provided",
    comments: str = "Not provided",
    conversation_summary: str = "Not provided"
):
    if not is_valid_email(user_email):
        return {"status": "failed", "reason": "invalid email"}

    subject = f"New website interaction from {user_name}"
    body = (
        f"New user interaction received.\n\n"
        f"User Name: {user_name}\n"
        f"User Email: {user_email}\n\n"
        f"Questions Asked:\n{questions_asked}\n\n"
        f"Comments:\n{comments}\n\n"
        f"Conversation Summary:\n{conversation_summary}\n"
    )

    send_email_via_smtp(subject, body)
    return {"status": "sent"}

email_user_interaction_json = {
    "name": "email_user_interaction",
    "description": (
        "Use this tool when a user wants to get in touch or discuss next steps "
        "and has provided their email address. "
        "Send the user's email, questions, comments, and a short summary of the conversation by email."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_email": {
                "type": "string",
                "description": "The user's email address"
            },
            "user_name": {
                "type": "string",
                "description": "The user's name if they provided it"
            },
            "questions_asked": {
                "type": "string",
                "description": "A concise list or summary of the user's questions"
            },
            "comments": {
                "type": "string",
                "description": "Any comments, interest, feedback, or intent expressed by the user"
            },
            "conversation_summary": {
                "type": "string",
                "description": "A short summary of the conversation"
            }
        },
        "required": ["user_email"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": email_user_interaction_json}]

tool_registry = {
    "email_user_interaction": email_user_interaction,
}

class Me:

    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("BASE_URL"))
        logger.info("Initializing Me agent")

        self.model_name = os.getenv("MODEL")
        if not self.model_name:
            raise ValueError("Missing MODEL environment variable. Set MODEL in your .env file.")
        logger.info("Using model: %s", self.model_name)

        self.name = "Hossein Masoudi"
        
        reader = PdfReader(str(BASE_DIR / "MyResume.pdf"))
        if reader is None or len(reader.pages) == 0:
            raise FileNotFoundError(f"Resume file not found: {reader}")
        logger.info("Resume loaded successfully with %d pages", len(reader.pages))

        self.resume = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.resume += text
        logger.debug("Resume text length: %d characters", len(self.resume))

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            logger.info("Tool called: %s", tool_name)
            tool = tool_registry.get(tool_name)
            if tool is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = tool(**arguments)
                except Exception as exc:
                    result = {"error": f"Tool {tool_name} failed: {exc}"}
            logger.info("Tool result: %s", result)
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results

    def normalize_history(self, history):
        logger.info("Normalizing conversation history")
        normalized = []
        for item in history:
            if isinstance(item, dict):
                normalized.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_message, assistant_message = item
                normalized.append({"role": "user", "content": user_message})
                if assistant_message is not None:
                    normalized.append({"role": "assistant", "content": assistant_message})
        logger.info("Conversation history normalized with %d messages", len(normalized))
        return normalized

    def system_prompt(self):
        return (
            f"You are acting as {self.name}.\n"
            f"You are answering questions on {self.name}'s website.\n"
            f"Focus on career topics: background, skills, experience, projects, and fit for roles.\n\n"

            f"Communication style:\n"
            f"- Be professional, warm, and engaging.\n"
            f"- Keep answers concise and helpful.\n"
            f"- Reply in the same language as the user.\n"
            f"- Do not invent facts.\n\n"

            f"Tool usage rules (must follow):\n"
            f"1) If the user expresses interest in contacting, hiring, collaborating, or continuing the conversation, ask them for their email address.\n"
            f"2) Only call `email_user_interaction` after the user explicitly provides their email address.\n"
            f"3) When calling `email_user_interaction`, summarize the user's questions and comments from the conversation so far.\n"
            f"4) Include the user's name if they provided it.\n"
            f"5) Never call the tool without an email address.\n\n"

            f"## Resume\n{self.resume}\n"
        )

    def chat(self, message, history):
        logger.info("Calling chat function with message: %s", message)
        messages = [{"role": "system", "content": self.system_prompt()}]
        messages.extend(self.normalize_history(history))
        messages.append({"role": "user", "content": message})
        done = False
        while not done:
            response = self.openai.chat.completions.create(model=self.model_name, messages=messages, tools=tools)
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
