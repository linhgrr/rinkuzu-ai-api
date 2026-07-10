from __future__ import annotations

import os

from locust import HttpUser, between, task


class ApiUser(HttpUser):
    wait_time = between(0.2, 0.8)

    def on_start(self):
        user_id = os.getenv("PERF_USER_ID")
        service_token = os.getenv("PERF_SERVICE_TOKEN")
        if user_id:
            self.client.headers["x-user-id"] = user_id
        if service_token:
            self.client.headers["x-service-token"] = service_token

    @task(4)
    def live(self):
        self.client.get("/api/live", name="GET /api/live")

    @task(2)
    def ready(self):
        self.client.get("/api/ready", name="GET /api/ready")

    @task(2)
    def pipeline_jobs(self):
        if os.getenv("PERF_USER_ID"):
            self.client.get("/api/v1/pipeline/jobs?limit=20", name="GET /pipeline/jobs")

    @task(1)
    def pipeline_job_status(self):
        job_id = os.getenv("PERF_PIPELINE_JOB_ID")
        if os.getenv("PERF_USER_ID") and job_id:
            self.client.get(f"/api/v1/pipeline/jobs/{job_id}", name="GET /pipeline/jobs/:id")
