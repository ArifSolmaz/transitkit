import pytest
from fastapi.testclient import TestClient
import json

from transitkit.api.main import app


class TestAPI:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    @pytest.mark.integration
    def test_fit_endpoint(self, client, synthetic_transit_light_curve):
        """Test fit endpoint with real data."""
        # First upload light curve
        light_curve_data = {
            "time": synthetic_transit_light_curve.time.tolist(),
            "flux": synthetic_transit_light_curve.flux.tolist(),
            "flux_err": synthetic_transit_light_curve.flux_err.tolist()
        }
        
        upload_response = client.post(
            "/lightcurves/upload",
            json=light_curve_data
        )
        assert upload_response.status_code == 200
        lc_id = upload_response.json()["id"]
        
        # Then fit
        fit_request = {
            "light_curve_id": lc_id,
            "parameters": {
                "t0": 15.0,
                "period": 5.0,
                "rp_over_rs": 0.1,
                "a_over_rs": 15.0,
                "inc": 89.0,
                "u1": 0.5,
                "u2": 0.0
            },
            "method": "least_squares"
        }
        
        response = client.post("/fit", json=fit_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert data["success"] is True
        assert "parameters" in data
        assert "chi2" in data