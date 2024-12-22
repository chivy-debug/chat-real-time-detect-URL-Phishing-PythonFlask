from dataclasses import dataclass
from typing import Dict, Set
from datetime import datetime
import threading


@dataclass
class User:
    username: str
    session_id: str
    login_time: datetime


class UserManager:
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, str] = {}  # session_id -> username
        self._lock = threading.Lock()

    def add_user(self, username: str, session_id: str) -> bool:
        with self._lock:
            if username in self._users:
                return False

            user = User(
                username=username,
                session_id=session_id,
                login_time=datetime.now()
            )
            self._users[username] = user
            self._sessions[session_id] = username
            return True

    def remove_user(self, username: str = None, session_id: str = None) -> bool:
        with self._lock:
            if username:
                if username not in self._users:
                    return False
                session_id = self._users[username].session_id
                del self._users[username]
                del self._sessions[session_id]
                return True
            elif session_id:
                if session_id not in self._sessions:
                    return False
                username = self._sessions[session_id]
                del self._users[username]
                del self._sessions[session_id]
                return True
            return False

    def get_user_by_session(self, session_id: str) -> User:
        with self._lock:
            username = self._sessions.get(session_id)
            return self._users.get(username) if username else None

    def get_active_users(self) -> Set[str]:
        with self._lock:
            return set(self._users.keys())

    def is_username_taken(self, username: str) -> bool:
        with self._lock:
            return username in self._users
