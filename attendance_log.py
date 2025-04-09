import json
import os
from datetime import datetime

LOG_FILE = "attendance_log.json"

def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_log(data):
    with open(LOG_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def log_attendance(name, date, time, photo_path):
    log = load_log()
    if date not in log:
        log[date] = []
    log[date].append({
        "name": name,
        "time": time,
        "photo": photo_path
    })
    save_log(log)