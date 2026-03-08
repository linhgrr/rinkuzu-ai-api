import asyncio
import os
import sys
from dotenv import load_dotenv

# Add parent dir to path so we can import api
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api/.env"))
    
    from motor.motor_asyncio import AsyncIOMotorClient
    uri = os.environ.get("MONGODB_URI")
    db_name = os.environ.get("MONGODB_DB_NAME", "adaptive_learning")
    client = AsyncIOMotorClient(uri)
    db = client[db_name]
    
    print("Listing latest pipeline jobs...")
    cursor = db["al_pipeline_jobs"].find({}, sort=[("completed_at", -1)]).limit(3)
    async for job in cursor:
        job_id = job.get("job_id")
        filename = job.get("filename", "")
        edges = job.get("result", {}).get("prereq_edges", [])
        concepts = job.get("result", {}).get("concept_map", {})
        print(f"[{job.get('created_at')}] Job: {job_id} | File: {filename} | Concepts: {len(concepts)} | Edges: {len(edges)}")

if __name__ == "__main__":
    asyncio.run(main())
