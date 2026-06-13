"""Qmatic Web Booking client for https://rezerwacja.lublin.eu/qmaticwebbooking/."""

import logging
from datetime import date
from difflib import get_close_matches

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://rezerwacja.lublin.eu/qmaticwebbooking/rest/schedule"
_TIMEOUT = 10.0


async def _get(path: str) -> list | dict:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{_BASE}{path}", headers={"Accept": "application/json"})
        resp.raise_for_status()
        return resp.json()


async def _get_services() -> list[dict]:
    return await _get("/services")  # type: ignore[return-value]


async def _find_service(name: str) -> dict | None:
    services = await _get_services()
    needle = name.lower().strip()
    for s in services:
        if s["name"].lower() == needle:
            return s
    for s in services:
        if needle in s["name"].lower() or s["name"].lower() in needle:
            return s
    names = [s["name"] for s in services]
    close = get_close_matches(name, names, n=1, cutoff=0.6)
    if close:
        return next(s for s in services if s["name"] == close[0])
    return None


async def get_reservation_slots(service_name: str, date_from: str | None = None) -> dict:
    """
    Return available reservation slots for the given city-hall service.

    Args:
        service_name: Polish service name, e.g. "Dowody osobiste".
        date_from: ISO date string (YYYY-MM-DD) to search from; defaults to today.

    Returns a dict with keys:
        service, duration_minutes, slots (list of {branch, address, date, available_times})
        or error / available_services on failure.
    """
    service = await _find_service(service_name)
    if not service:
        all_services = await _get_services()
        return {
            "error": f"Nie znaleziono usługi '{service_name}'.",
            "available_services": [s["name"] for s in all_services],
        }

    sid = service["publicId"]

    branches_raw = await _get(f"/branches/available;servicePublicId={sid}")
    branches: list[dict] = (
        branches_raw.get("value", []) if isinstance(branches_raw, dict) else branches_raw  # type: ignore[union-attr]
    )
    if not branches:
        return {
            "error": f"Brak dostępnych lokalizacji dla usługi '{service['name']}'.",
            "service": service["name"],
        }

    since = date_from or date.today().isoformat()
    result: dict = {
        "service": service["name"],
        "duration_minutes": service["duration"],
        "booking_url": "https://rezerwacja.lublin.eu/qmaticwebbooking/#/",
        "slots": [],
    }

    for branch in branches[:3]:
        bid = branch["id"]
        try:
            raw_dates: list[dict] = await _get(f"/branches/{bid}/dates;servicePublicId={sid}")  # type: ignore[assignment]
            upcoming = [d["date"] for d in raw_dates if d["date"] >= since][:5]
        except Exception as exc:
            logger.warning("dates fetch failed for branch %s: %s", branch["name"], exc)
            continue

        for date_str in upcoming:
            try:
                raw_times: list[dict] = await _get(f"/branches/{bid}/dates/{date_str}/times;servicePublicId={sid}")  # type: ignore[assignment]
                times = [t["time"] for t in raw_times]
            except Exception as exc:
                logger.warning("times fetch failed %s %s: %s", branch["name"], date_str, exc)
                continue

            if times:
                result["slots"].append(
                    {
                        "branch": branch["name"],
                        "address": branch.get("addressLine1", ""),
                        "date": date_str,
                        "available_times": times,
                    }
                )

    if not result["slots"]:
        result["message"] = "Brak dostępnych terminów w najbliższym czasie."

    return result
