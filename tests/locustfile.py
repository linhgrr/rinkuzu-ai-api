from __future__ import annotations

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    wait_time = between(1, 2)

    @task(4)
    def ready(self):
        self.client.get("/api/ready")

    @task(2)
    def quiz_drafts(self):
        self.client.get("/api/v1/quiz/drafts")

    @task(1)
    def session_status(self):
        self.client.get("/api/v1/session/status")

    @task(1)
    def pipeline_status(self):
        self.client.get("/api/v1/pipeline/status")
