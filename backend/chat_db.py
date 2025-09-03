import sqlite3
from datetime import datetime
from textblob import TextBlob

DB_PATH = 'chat_history.db'

# Create table if not exists
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_id TEXT,
    question TEXT,
    answer TEXT,
    sentiment TEXT
)
''')
conn.commit()
conn.close()

def compute_sentiment(text):
    """Return polarity as string: positive, neutral, negative."""
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.15:
            return "positive"
        elif polarity < -0.15:
            return "negative"
        else:
            return "neutral"
    except Exception:
        return "unknown"

def save_chat_history(user_id, question, answer, sentiment=None):
    if sentiment is None:
        sentiment = compute_sentiment(answer)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (timestamp, user_id, question, answer, sentiment)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.utcnow().isoformat(), user_id, question, answer, sentiment))
    conn.commit()
    conn.close()

def get_chat_history(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT timestamp, user_id, question, answer, sentiment FROM chat_history ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [
        {
            'timestamp': r[0],
            'user_id': r[1],
            'question': r[2],
            'answer': r[3],
            'sentiment': r[4]
        } for r in rows
    ]
