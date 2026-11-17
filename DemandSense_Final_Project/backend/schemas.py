"""Request bodies for the API (response bodies are plain JSON)."""
from datetime import date as Date
from typing import List, Optional

from pydantic import BaseModel, Field


class SaleEntry(BaseModel):
    item: str = Field(min_length=1)
    kg: float = Field(ge=0, le=100000)


class DailySales(BaseModel):
    date: Date
    entries: List[SaleEntry]
    note: Optional[str] = ""
    retrain: bool = False


class PlannerRequest(BaseModel):
    item: str
    stock_kg: float = Field(ge=0)
    days: int = Field(default=7, ge=1, le=60)
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)
    shelf_life_days: Optional[int] = Field(default=None, ge=1, le=90)
