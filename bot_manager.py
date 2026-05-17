import asyncio
import json
import os
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import discord
import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

DB_PATH = Path("bots.db")
CONFIG_PATH = Path("bots_config.json")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-70b-versatile"
MEMORY_WINDOW = 12


@dataclass
class BotConfig:
    bot_id: int
    name: str
    token: str = ""
    prompt: str = "You are a helpful Discord assistant."
    enabled: bool = False


@dataclass
class BotState:
    config: BotConfig
    client: Optional[discord.Client] = None
    memory: Dict[int, deque] = field(default_factory=dict)


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id INTEGER,
                channel_id INTEGER,
                author TEXT,
                role TEXT,
                content TEXT,
                important INTEGER DEFAULT 0,
                created_at TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def add_message(self, bot_id: int, channel_id: int, author: str, role: str, content: str, important: bool = False):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO messages (bot_id, channel_id, author, role, content, important, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bot_id,
                channel_id,
                author,
                role,
                content,
                int(important),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    def get_messages(self, bot_id: int, limit: int = 100):
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT channel_id, author, role, content, important, created_at
            FROM messages
            WHERE bot_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (bot_id, limit),
        )
        rows = cur.fetchall()
        conn.close()
        return [
            {
                "channel_id": r[0],
                "author": r[1],
                "role": r[2],
                "content": r[3],
                "important": bool(r[4]),
                "created_at": r[5],
            }
            for r in rows
        ]


class DiscordBotManager:
    def __init__(self):
        self.storage = Storage(DB_PATH)
        self.bot_states: Dict[int, BotState] = {}
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.loop = asyncio.new_event_loop()
        self._load_configs()

    def _load_configs(self):
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text())
        else:
            data = {
                "bots": [
                    {"bot_id": i, "name": f"bot-{i}", "token": "", "prompt": "You are a helpful Discord assistant.", "enabled": False}
                    for i in range(1, 6)
                ]
            }
            CONFIG_PATH.write_text(json.dumps(data, indent=2))

        for cfg in data["bots"]:
            config = BotConfig(**cfg)
            self.bot_states[config.bot_id] = BotState(config=config)

    def save_configs(self):
        payload = {
            "bots": [
                {
                    "bot_id": state.config.bot_id,
                    "name": state.config.name,
                    "token": state.config.token,
                    "prompt": state.config.prompt,
                    "enabled": state.config.enabled,
                }
                for state in self.bot_states.values()
            ]
        }
        CONFIG_PATH.write_text(json.dumps(payload, indent=2))

    def _is_important(self, text: str) -> bool:
        lowered = text.lower()
        keywords = ["deadline", "password", "todo", "important", "remember", "meeting", "due", "address", "phone"]
        return any(k in lowered for k in keywords) or len(text) > 300

    def _get_memory(self, state: BotState, channel_id: int) -> deque:
        if channel_id not in state.memory:
            state.memory[channel_id] = deque(maxlen=MEMORY_WINDOW)
        return state.memory[channel_id]

    def _should_respond(self, msg: discord.Message) -> bool:
        if msg.author.bot:
            return False
        mentioned = msg.guild and msg.guild.me and msg.guild.me in msg.mentions
        return mentioned or msg.reference is not None

    def _build_prompt(self, state: BotState, channel_id: int, username: str, content: str) -> List[dict]:
        memory = self._get_memory(state, channel_id)
        msgs = [{"role": "system", "content": state.config.prompt}]
        msgs.extend(list(memory))
        msgs.append({"role": "user", "content": f"{username}: {content}"})
        return msgs

    def _groq_reply(self, messages: List[dict]) -> str:
        if not self.groq_api_key:
            return "Set GROQ_API_KEY in your environment so I can reply."

        response = requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 300,
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    async def _start_single_bot(self, state: BotState):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.messages = True

        client = discord.Client(intents=intents)
        state.client = client

        @client.event
        async def on_ready():
            print(f"[{state.config.name}] Logged in as {client.user}")

        @client.event
        async def on_message(message: discord.Message):
            if message.author == client.user:
                return

            if not self._should_respond(message):
                return

            memory = self._get_memory(state, message.channel.id)
            memory.append({"role": "user", "content": f"{message.author.display_name}: {message.content}"})

            important = self._is_important(message.content)
            self.storage.add_message(
                state.config.bot_id,
                message.channel.id,
                str(message.author),
                "user",
                message.content,
                important,
            )

            prompt = self._build_prompt(state, message.channel.id, message.author.display_name, message.content)
            try:
                reply = await asyncio.to_thread(self._groq_reply, prompt)
            except Exception as exc:
                reply = f"Failed to generate reply: {exc}"

            memory.append({"role": "assistant", "content": reply})
            self.storage.add_message(state.config.bot_id, message.channel.id, state.config.name, "assistant", reply)

            await message.reply(reply, mention_author=False)

        await client.start(state.config.token)

    async def launch_all(self):
        tasks = []
        for state in self.bot_states.values():
            if state.config.enabled and state.config.token:
                tasks.append(asyncio.create_task(self._start_single_bot(state)))
        if tasks:
            await asyncio.gather(*tasks)

    async def send_as_bot(self, bot_id: int, channel_id: int, text: str):
        state = self.bot_states[bot_id]
        if not state.client:
            raise RuntimeError("Bot is not connected")
        channel = state.client.get_channel(channel_id)
        if channel is None:
            channel = await state.client.fetch_channel(channel_id)
        await channel.send(text)
        self.storage.add_message(bot_id, channel_id, state.config.name, "assistant", text)

    def list_channels(self, bot_id: int):
        state = self.bot_states[bot_id]
        client = state.client
        if not client:
            return []
        channels = []
        for guild in client.guilds:
            for ch in guild.text_channels:
                if ch.permissions_for(guild.me).send_messages:
                    channels.append({"guild": guild.name, "channel": ch.name, "channel_id": ch.id})
        return channels


manager = DiscordBotManager()
app = Flask(__name__)
CORS(app)


@app.get("/")
def dashboard():
    return send_from_directory(".", "dashboard.html")



@app.get("/api/bots")
def get_bots():
    return jsonify(
        [
            {
                "bot_id": s.config.bot_id,
                "name": s.config.name,
                "token": s.config.token,
                "prompt": s.config.prompt,
                "enabled": s.config.enabled,
                "channels": manager.list_channels(s.config.bot_id),
            }
            for s in manager.bot_states.values()
        ]
    )


@app.post("/api/bots/<int:bot_id>")
def update_bot(bot_id: int):
    state = manager.bot_states[bot_id]
    payload = request.json or {}
    for key in ["name", "token", "prompt", "enabled"]:
        if key in payload:
            setattr(state.config, key, payload[key])
    manager.save_configs()
    return jsonify({"ok": True})


@app.get("/api/bots/<int:bot_id>/messages")
def get_messages(bot_id: int):
    limit = int(request.args.get("limit", 100))
    return jsonify(manager.storage.get_messages(bot_id, limit=limit))


@app.post("/api/bots/<int:bot_id>/send")
def send_message(bot_id: int):
    payload = request.json or {}
    channel_id = int(payload["channel_id"])
    text = payload["message"]
    fut = asyncio.run_coroutine_threadsafe(manager.send_as_bot(bot_id, channel_id, text), manager.loop)
    fut.result(timeout=20)
    return jsonify({"ok": True})


def main():
    import threading

    def run_bots():
        asyncio.set_event_loop(manager.loop)
        manager.loop.run_until_complete(manager.launch_all())

    thread = threading.Thread(target=run_bots, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
