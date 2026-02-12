"""
End-to-End Validation Script for Argus Core Platform
Validates all infrastructure components and API endpoints
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from minio import Minio
from redis import Redis
from motor.motor_asyncio import AsyncIOMotorClient
import httpx

# Configuration
BACKEND_URL = "http://localhost:8001"
MONGO_URL = "mongodb://localhost:27017"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

class ValidationReport:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, status: bool, details: str = ""):
        self.tests.append({
            "name": name,
            "status": "✅ PASS" if status else "❌ FAIL",
            "details": details
        })
        if status:
            self.passed += 1
        else:
            self.failed += 1
    
    def print_report(self):
        print("\n" + "="*80)
        print("ARGUS CORE - END-TO-END VALIDATION REPORT")
        print("="*80 + "\n")
        
        for test in self.tests:
            print(f"{test['status']} {test['name']}")
            if test['details']:
                print(f"   └─ {test['details']}")
        
        print("\n" + "="*80)
        print(f"SUMMARY: {self.passed} passed, {self.failed} failed, {len(self.tests)} total")
        print("="*80 + "\n")
        
        return self.failed == 0


async def validate_infrastructure():
    """Validate all infrastructure components"""
    report = ValidationReport()
    
    # Test 1: Redis Connectivity
    try:
        redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, socket_connect_timeout=5)
        result = redis_client.ping()
        report.add_test("Redis Connectivity", result, f"PONG received from {REDIS_HOST}:{REDIS_PORT}")
        redis_client.close()
    except Exception as e:
        report.add_test("Redis Connectivity", False, str(e))
    
    # Test 2: MongoDB Connectivity
    try:
        mongo_client = AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        await mongo_client.admin.command('ping')
        db_list = await mongo_client.list_database_names()
        report.add_test("MongoDB Connectivity", True, f"Connected, found {len(db_list)} databases")
        mongo_client.close()
    except Exception as e:
        report.add_test("MongoDB Connectivity", False, str(e))
    
    # Test 3: MinIO Connectivity
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        buckets = minio_client.list_buckets()
        bucket_names = [b.name for b in buckets]
        report.add_test("MinIO Connectivity", True, f"Connected, buckets: {bucket_names}")
    except Exception as e:
        report.add_test("MinIO Connectivity", False, str(e))
    
    # Test 4: Backend API Health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/health", timeout=10.0)
            status = response.json().get("status") == "healthy"
            report.add_test("Backend API Health", status, f"Status code: {response.status_code}")
    except Exception as e:
        report.add_test("Backend API Health", False, str(e))
    
    # Test 5: Backend Detailed Health Check
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/health", timeout=10.0)
            health_data = response.json()
            status = health_data.get("status") == "healthy"
            components = health_data.get("components", {})
            
            details = f"Database: {components.get('database')}, Storage: {components.get('storage')}"
            if 'models' in components:
                models = components['models']
                details += f", Models loaded: {models.get('loaded', 0)}/{6}"
            
            report.add_test("Backend Detailed Health", status, details)
    except Exception as e:
        report.add_test("Backend Detailed Health", False, str(e))
    
    # Test 6: API Root Endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/", timeout=10.0)
            data = response.json()
            status = response.status_code == 200 and "Argus Core" in data.get("name", "")
            report.add_test("API Root Endpoint", status, f"API Version: {data.get('version')}")
    except Exception as e:
        report.add_test("API Root Endpoint", False, str(e))
    
    # Test 7: Models Endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/models", timeout=10.0)
            data = response.json()
            models = data.get("models", {})
            status = response.status_code == 200 and len(models) > 0
            report.add_test("Models Endpoint", status, f"Available models: {len(models)}")
    except Exception as e:
        report.add_test("Models Endpoint", False, str(e))
    
    # Test 8: Stats Endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/api/v1/stats", timeout=10.0)
            data = response.json()
            status = response.status_code == 200 and "total" in data
            report.add_test("Stats Endpoint", status, f"Total analyses: {data.get('total', 0)}")
    except Exception as e:
        report.add_test("Stats Endpoint", False, str(e))
    
    # Test 9: OpenAPI Documentation
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/openapi.json", timeout=10.0)
            data = response.json()
            paths = list(data.get("paths", {}).keys())
            status = response.status_code == 200 and len(paths) > 0
            report.add_test("OpenAPI Documentation", status, f"Available endpoints: {len(paths)}")
    except Exception as e:
        report.add_test("OpenAPI Documentation", False, str(e))
    
    # Test 10: Frontend Availability
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:3000", timeout=10.0)
            status = response.status_code == 200 and "Argus Core" in response.text
            report.add_test("Frontend Availability", status, f"Status code: {response.status_code}")
    except Exception as e:
        report.add_test("Frontend Availability", False, str(e))
    
    return report


async def main():
    print("\n🚀 Starting Argus Core End-to-End Validation...\n")
    report = await validate_infrastructure()
    success = report.print_report()
    
    if success:
        print("🎉 ALL TESTS PASSED! System is fully operational.\n")
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED! Review the report above.\n")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
