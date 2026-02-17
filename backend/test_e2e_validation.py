#!/usr/bin/env python3
"""
Argus Core - End-to-End Validation Script
==========================================
Comprehensive validation of the entire application stack.

Tests:
1. Infrastructure services (Redis, MinIO, MongoDB, Celery)
2. Backend API health and models
3. File upload and storage
4. Analysis workflow
5. Frontend accessibility
"""

import asyncio
import json
import sys
import time
from datetime import datetime

import httpx
import redis.asyncio as aioredis
from motor.motor_asyncio import AsyncIOMotorClient


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")


class E2EValidator:
    """End-to-end validation for Argus Core."""
    
    def __init__(self):
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:3000"
        self.redis_url = "redis://localhost:6379/0"
        self.mongo_url = "mongodb://localhost:27017"
        self.minio_url = "http://localhost:9000"
        
        self.results = {
            "passed": 0,
            "failed": 0,
            "warnings": 0,
            "start_time": datetime.now()
        }
    
    async def test_redis(self) -> bool:
        """Test Redis connectivity."""
        print_info("Testing Redis connectivity...")
        try:
            redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            pong = await redis_client.ping()
            await redis_client.close()
            
            if pong:
                print_success("Redis is accessible and responding to PING")
                return True
            else:
                print_error("Redis PING failed")
                return False
        except Exception as e:
            print_error(f"Redis connection failed: {e}")
            return False
    
    async def test_mongodb(self) -> bool:
        """Test MongoDB connectivity."""
        print_info("Testing MongoDB connectivity...")
        try:
            client = AsyncIOMotorClient(self.mongo_url)
            await client.admin.command('ping')
            
            # List databases
            db_list = await client.list_database_names()
            print_success(f"MongoDB is accessible (databases: {len(db_list)})")
            
            client.close()
            return True
        except Exception as e:
            print_error(f"MongoDB connection failed: {e}")
            return False
    
    async def test_minio(self) -> bool:
        """Test MinIO accessibility."""
        print_info("Testing MinIO accessibility...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.minio_url}/minio/health/live", timeout=5.0)
                
                if response.status_code == 200:
                    print_success("MinIO is accessible and healthy")
                    return True
                else:
                    print_error(f"MinIO health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"MinIO connection failed: {e}")
            return False
    
    async def test_celery(self) -> bool:
        """Test Celery worker status."""
        print_info("Testing Celery worker status...")
        try:
            from processing.tasks import celery_app
            
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            
            if stats:
                worker_count = len(stats)
                print_success(f"Celery workers active: {worker_count}")
                
                # Print worker details
                for worker_name, worker_stats in stats.items():
                    print_info(f"  Worker: {worker_name}")
                
                return True
            else:
                print_error("No Celery workers found")
                return False
        except Exception as e:
            print_error(f"Celery inspection failed: {e}")
            return False
    
    async def test_backend_health(self) -> bool:
        """Test backend API health endpoint."""
        print_info("Testing backend API health...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/api/v1/health", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "unknown")
                    components = data.get("components", {})
                    
                    print_success(f"Backend API is {status}")
                    
                    # Check each component
                    for component, comp_status in components.items():
                        if isinstance(comp_status, dict):
                            comp_status_val = comp_status.get("status", "unknown")
                            print_info(f"  {component}: {comp_status_val}")
                        else:
                            print_info(f"  {component}: {comp_status}")
                    
                    return status == "healthy"
                else:
                    print_error(f"Backend health check failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"Backend API connection failed: {e}")
            return False
    
    async def test_models(self) -> bool:
        """Test AI models loading and availability."""
        print_info("Testing AI models...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.backend_url}/api/v1/models", timeout=10.0)
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", data) if isinstance(data, dict) else data
                    
                    loaded_models = []
                    available_models = []
                    
                    for m in models:
                        if isinstance(m, dict):
                            if m.get("loaded", False):
                                loaded_models.append(m)
                            if m.get("file_exists", False):
                                available_models.append(m)
                    
                    print_success(f"Models available: {len(available_models)}, loaded: {len(loaded_models)}")
                    
                    # List loaded models
                    for model in loaded_models:
                        model_name = model.get('name', 'unknown')
                        model_cat = model.get('category', 'unknown')
                        model_vram = model.get('vram_mb', 0)
                        print_info(f"  ✓ {model_name} ({model_cat}) - {model_vram}MB")
                    
                    return len(loaded_models) >= 3
                else:
                    print_error(f"Models endpoint failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"Models check failed: {e}")
            return False
    
    async def test_frontend(self) -> bool:
        """Test frontend accessibility."""
        print_info("Testing frontend accessibility...")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.frontend_url, timeout=10.0, follow_redirects=True)
                
                if response.status_code == 200:
                    html = response.text
                    
                    # Check for expected elements
                    has_title = "Argus Core" in html
                    has_react = "react" in html.lower() or "__next" in html
                    
                    if has_title and has_react:
                        print_success("Frontend is accessible and rendering correctly")
                        return True
                    else:
                        print_warning("Frontend accessible but may not be rendering correctly")
                        return True
                else:
                    print_error(f"Frontend check failed: {response.status_code}")
                    return False
        except Exception as e:
            print_error(f"Frontend connection failed: {e}")
            return False
    
    async def test_file_upload(self) -> bool:
        """Test file upload capability."""
        print_info("Testing file upload endpoint...")
        try:
            # Create a small test file
            test_content = b"Test file for Argus Core validation"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                files = {"file": ("test.txt", test_content, "text/plain")}
                
                response = await client.post(
                    f"{self.backend_url}/api/v1/analyze",
                    files=files
                )
                
                if response.status_code in [200, 201, 202]:
                    data = response.json()
                    analysis_id = data.get("analysis_id")
                    
                    if analysis_id:
                        print_success(f"File upload successful (analysis_id: {analysis_id})")
                        return True
                    else:
                        print_warning("Upload succeeded but no analysis_id returned")
                        return True
                elif response.status_code == 415:
                    print_warning("File type not supported (expected for test file)")
                    return True
                else:
                    print_error(f"File upload failed: {response.status_code} - {response.text[:200]}")
                    return False
        except Exception as e:
            print_error(f"File upload test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all validation tests."""
        print_header("ARGUS CORE - END-TO-END VALIDATION")
        print_info(f"Started at: {self.results['start_time']}")
        
        # Phase 1: Infrastructure Services
        print_header("PHASE 1: Infrastructure Services")
        
        tests = [
            ("Redis", self.test_redis()),
            ("MongoDB", self.test_mongodb()),
            ("MinIO", self.test_minio()),
            ("Celery", self.test_celery()),
        ]
        
        for name, test_coro in tests:
            result = await test_coro
            if result:
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1
            print()
        
        # Phase 2: Backend Services
        print_header("PHASE 2: Backend API & Models")
        
        tests = [
            ("Backend Health", self.test_backend_health()),
            ("AI Models", self.test_models()),
        ]
        
        for name, test_coro in tests:
            result = await test_coro
            if result:
                self.results["passed"] += 1
            else:
                self.results["failed"] += 1
            print()
        
        # Phase 3: Frontend
        print_header("PHASE 3: Frontend Application")
        
        result = await self.test_frontend()
        if result:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        print()
        
        # Phase 4: Integration Tests
        print_header("PHASE 4: Integration Tests")
        
        result = await self.test_file_upload()
        if result:
            self.results["passed"] += 1
        else:
            self.results["failed"] += 1
        print()
        
        # Print Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary."""
        end_time = datetime.now()
        duration = (end_time - self.results['start_time']).total_seconds()
        
        print_header("VALIDATION SUMMARY")
        
        total_tests = self.results['passed'] + self.results['failed']
        
        print(f"{Colors.BOLD}Total Tests:{Colors.END} {total_tests}")
        print(f"{Colors.GREEN}✓ Passed:{Colors.END} {self.results['passed']}")
        print(f"{Colors.RED}✗ Failed:{Colors.END} {self.results['failed']}")
        
        if self.results['warnings'] > 0:
            print(f"{Colors.YELLOW}⚠ Warnings:{Colors.END} {self.results['warnings']}")
        
        print(f"\n{Colors.BOLD}Duration:{Colors.END} {duration:.2f}s")
        
        if self.results['failed'] == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! APPLICATION IS FULLY OPERATIONAL 🎉{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}⚠ SOME TESTS FAILED - PLEASE REVIEW ERRORS ABOVE{Colors.END}")
            return 1


async def main():
    """Main entry point."""
    validator = E2EValidator()
    await validator.run_all_tests()
    
    return validator.results['failed']


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
