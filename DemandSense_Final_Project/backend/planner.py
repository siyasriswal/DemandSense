"""
Stock planner: turns a demand forecast into a purchase suggestion and a wastage warning.

Assumptions (shown to the user in the dashboard):
  * safety stock = z x standard deviation of past forecast errors x sqrt(days)
  * stock on hand has to be sold within the item's shelf life, otherwise it is wasted
"""
import math

import pandas as pd

SERVICE_Z = {0.80: 0.84, 0.90: 1.28, 0.95: 1.645, 0.99: 2.33}


def plan(forecast: pd.DataFrame, error_std: float, stock_kg: float, shelf_life_days: int, service_level: float = 0.95,
         days: int = None):
    """`forecast` must cover at least max(days, shelf_life_days) days; `days` is the planning horizon."""
    z = SERVICE_Z.get(round(service_level, 2), 1.645)
    days = min(days or len(forecast), len(forecast))
    expected_total = float(forecast.head(days)["forecast"].sum())

    # Perishable goods are bought in batches: one batch has to last about as long as the shelf life.
    cycle = max(1, min(days, shelf_life_days))
    cycle_demand = float(forecast.head(cycle)["forecast"].sum())
    safety = z * error_std * math.sqrt(cycle)
    target = cycle_demand + safety
    order = max(0.0, target - stock_kg)

    shelf_demand = float(forecast.head(max(1, min(len(forecast), shelf_life_days)))["forecast"].sum())
    spoil = max(0.0, stock_kg - shelf_demand)          # stock that will not sell before it expires
    spoil_pct = spoil / stock_kg * 100 if stock_kg > 0 else 0.0
    risk = "Low" if spoil_pct < 10 else "Medium" if spoil_pct < 30 else "High"

    cumulative = forecast["forecast"].cumsum()
    remaining = (stock_kg - cumulative).clip(lower=0)
    sold_out = forecast.loc[cumulative >= stock_kg, "date"]
    cover_days = int((sold_out.iloc[0] - forecast["date"].iloc[0]).days) + 1 if len(sold_out) else None

    notes = [f"Buy for {cycle} day(s) at a time, because this item lasts about {shelf_life_days} day(s). "
             f"Target stock for the next {cycle} day(s): {target:,.0f} kg."]
    if order > 0:
        notes.append(f"Order about {order:,.0f} kg now.")
    else:
        notes.append("Current stock already covers the next batch, including the safety margin. No order needed now.")
    if cover_days is not None:
        notes.append(f"At the forecast rate, current stock lasts about {cover_days} day(s).")
    if spoil > 0:
        notes.append(f"About {spoil:,.0f} kg may not sell before it expires ({spoil_pct:.0f}% of stock).")
    festivals = sorted({f for f in forecast["festival"].dropna()})
    if festivals:
        notes.append("Festival demand in this period: " + ", ".join(festivals) + ".")

    daily = pd.DataFrame({"date": forecast["date"].dt.strftime("%Y-%m-%d"), "forecast": forecast["forecast"],
                          "cumulative_demand": cumulative.round(1), "remaining_stock": remaining.round(1)})
    return {
        "days": days, "cycle_days": cycle, "cycle_demand_kg": round(cycle_demand, 1),
        "expected_demand_kg": round(expected_total, 1), "safety_stock_kg": round(safety, 1),
        "target_stock_kg": round(target, 1), "recommended_order_kg": round(order, 1),
        "shelf_life_days": shelf_life_days, "expected_spoilage_kg": round(spoil, 1),
        "spoilage_pct": round(spoil_pct, 1), "wastage_risk": risk, "stock_cover_days": cover_days,
        "service_level": service_level, "notes": notes, "daily": daily.to_dict("records"),
    }
