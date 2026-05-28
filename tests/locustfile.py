"""
Locust load test for NeuroSync backend.
Run with: locust -f tests/locustfile.py --users 10 --spawn-rate 2 --run-time 5m
Requires the server to be running on localhost:8000.
"""
from locust import HttpUser, task, between


class NeuroUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def design_and_start(self):
        r = self.client.post("/api/session/design", json={
            "user_text": "I feel stressed, want calm for 5 minutes.",
            "user_id": "load_test_user",
            "client_type": "mobile",
        })
        if r.status_code == 200 and r.json().get("blueprint"):
            self.client.post("/api/session/start", json=r.json()["blueprint"])
