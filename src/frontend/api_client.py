import requests

API_BASE_URL = "http://localhost:8000"

def send_query(query: str, session_id: str):
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"query": query, "session_id": session_id}
    )
    return response.json()

def send_feedback(row_id: int, rating: int):
    response = requests.post(
        f"{API_BASE_URL}/feedback",
        json={"row_id": row_id, "rating": rating}
    )
    return response.json()

def fetch_analytics():
    response = requests.get(f"{API_BASE_URL}/analytics")
    return response.json()