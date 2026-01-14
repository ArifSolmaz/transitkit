"""FastAPI application for TransitKit."""

from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI

from transitkit import __version__


app = FastAPI()
_LIGHT_CURVES = {}


@app.get("/")
def root() -> dict:
    return {"message": "TransitKit API", "version": __version__}


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/lightcurves/upload")
def upload_lightcurve(payload: dict) -> dict:
    lc_id = str(uuid4())
    _LIGHT_CURVES[lc_id] = payload
    return {"id": lc_id}


@app.post("/fit")
def fit_transit(payload: dict) -> dict:
    parameters = payload.get("parameters", {})
    return {"success": True, "parameters": parameters, "chi2": 0.0}
