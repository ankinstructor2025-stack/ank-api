import hashlib

def user_queue_name(uid: str) -> str:
    h = hashlib.sha1(uid.encode()).hexdigest()[:16]
    return f"q-{h}"
