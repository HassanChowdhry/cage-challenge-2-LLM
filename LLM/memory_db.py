import sqlite3
import json
from typing import Iterable, Tuple


class TrajectoryDatabase:
    """Simple SQLite backend to store agent transitions."""

    def __init__(self, path: str = "trajectories.db") -> None:
        self.conn = sqlite3.connect(path)
        self._setup()

    def _setup(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS transitions("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "state TEXT,"
            "action TEXT,"
            "reward REAL"
            ")"
        )
        self.conn.commit()

    def insert(self, state, action, reward: float) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO transitions(state, action, reward) VALUES (?, ?, ?)",
            (json.dumps(state), json.dumps(action), float(reward)),
        )
        self.conn.commit()

    def all_transitions(self) -> Iterable[Tuple[dict, dict, float]]:
        cur = self.conn.cursor()
        cur.execute("SELECT state, action, reward FROM transitions")
        for s, a, r in cur.fetchall():
            yield json.loads(s), json.loads(a), r

    def close(self) -> None:
        self.conn.close()
