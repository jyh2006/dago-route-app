from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Tuple

import math
import requests
from itertools import permutations

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# -----------------------------------------------------------------------------
# FastAPI setup
# -----------------------------------------------------------------------------

app = FastAPI(title="Multi-stop Route Optimizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Pydantic models
# -----------------------------------------------------------------------------

class OptimizeByNameRequest(BaseModel):
    lines: List[str]
    start_index: int = 0
    end_index: Optional[int] = None


class Location(BaseModel):
    lat: float
    lon: float


# -----------------------------------------------------------------------------
# Distance helpers (straight-line model)
# -----------------------------------------------------------------------------

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great-circle distance in kilometers between two (lat, lon) points."""
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def build_distance_matrix(locations: List[Location]) -> List[List[int]]:
    """
    Return an NxN matrix of travel cost (seconds), using straight-line distance
    and an average speed of ~30 km/h.
    """
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i][j] = 0
            else:
                d_km = haversine_km(
                    (locations[i].lat, locations[i].lon),
                    (locations[j].lat, locations[j].lon),
                )
                t_hours = d_km / 30.0  # 30 km/h
                matrix[i][j] = int(t_hours * 3600)
    return matrix


# -----------------------------------------------------------------------------
# TSP solver (exact for small n, OR-Tools for large n)
# -----------------------------------------------------------------------------

def solve_tsp(locations: List[Location], start_index: int, end_index: Optional[int] = None):
    """
    Return a route visiting ALL points exactly once.

    - If end_index is None or equal to start_index: closed loop (start=end).
    - Otherwise: open path (start != end).

    For n <= 9: exact brute-force search (guaranteed optimal).
    For n > 9:  OR-Tools with local search (good heuristic).
    """
    n = len(locations)
    if n < 2:
        return [start_index]

    dist_matrix = build_distance_matrix(locations)

    def route_cost(route: List[int]) -> int:
        return sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

    # ----- SMALL n: exact brute force -----
    if n <= 9:
        indices = list(range(n))

        # closed loop: start == end
        if end_index is None or end_index == start_index:
            middle = [i for i in indices if i != start_index]
            best_route = None
            best_cost = float("inf")
            for perm in permutations(middle):
                cand = [start_index] + list(perm) + [start_index]
                c = route_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_route = cand
            return best_route

        # open path: start != end
        middle = [i for i in indices if i not in (start_index, end_index)]
        best_route = None
        best_cost = float("inf")
        for perm in permutations(middle):
            cand = [start_index] + list(perm) + [end_index]
            c = route_cost(cand)
            if c < best_cost:
                best_cost = c
                best_route = cand
        return best_route

    # ----- LARGE n: OR-Tools heuristic -----
    if end_index is None or end_index == start_index:
        manager = pywrapcp.RoutingIndexManager(n, 1, start_index)
    else:
        manager = pywrapcp.RoutingIndexManager(n, 1, [start_index], [end_index])

    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    # Improve the first solution via local search
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(10)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        raise RuntimeError("No solution found by OR-Tools.")

    route_indices: List[int] = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route_indices.append(node)
        index = solution.Value(routing.NextVar(index))
    route_indices.append(manager.IndexToNode(index))  # final node

    return route_indices


def optimize_core(locations: List[Location], start_index: int, end_index: Optional[int]):
    n = len(locations)
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 locations.")
    if not (0 <= start_index < n):
        raise HTTPException(status_code=400, detail="Invalid start_index.")
    if end_index is not None and not (0 <= end_index < n):
        raise HTTPException(status_code=400, detail="Invalid end_index.")

    route_indices = solve_tsp(locations, start_index, end_index)

    dist_matrix = build_distance_matrix(locations)
    total_time_seconds = sum(
        dist_matrix[route_indices[i]][route_indices[i + 1]]
        for i in range(len(route_indices) - 1)
    )
    total_time_hours = total_time_seconds / 3600.0

    solver_mode = "exact" if n <= 9 else "ortools"

    return {
        "route_indices": route_indices,
        "total_time_seconds": total_time_seconds,
        "total_time_hours": total_time_hours,
        "solver_mode": solver_mode,
        "solver_cost": total_time_seconds,  # same units as our cost
    }


# -----------------------------------------------------------------------------
# Geocoding via Nominatim (OpenStreetMap)
# -----------------------------------------------------------------------------

# at top of app.py
KAKAO_REST_API_KEY = "4c7b220b34f3fca0fe29501380a7262d"  # just the key, no "KakaoAK "

GEOCODE_CACHE: Dict[str, Tuple[float, float, str]] = {}


def geocode_line(raw_line: str) -> Tuple[float, float, str]:
    """
    Interpret a line as either:
      - 'lat,lon' (optionally with inline comments after '#')
      - a place name, which we geocode via Kakao Local Search API.
    Returns (lat, lon, label).
    """
    # 1) Clean line
    line = raw_line.split("#")[0].strip()
    if not line:
        raise HTTPException(status_code=400, detail="Empty line in input.")

    # 2) Try 'lat,lon' directly (bypass Kakao)
    if "," in line:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                label = raw_line.strip()
                return lat, lon, label
            except ValueError:
                pass  # fall through to name geocoding

    # 3) Cache
    if line in GEOCODE_CACHE:
        lat, lon, label = GEOCODE_CACHE[line]
        return lat, lon, label

    # 4) Safety: key must be set and not include 'KakaoAK '
    if not KAKAO_REST_API_KEY or KAKAO_REST_API_KEY.startswith("YOUR_"):
        raise HTTPException(
            status_code=500,
            detail="Kakao REST API key not set in server. Please configure KAKAO_REST_API_KEY.",
        )

    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}",
    }

    def kakao_search(query: str) -> dict:
        params = {"query": query, "size": 1}
        try:
            r = requests.get(url, headers=headers, params=params, timeout=5)
        except requests.RequestException as e:
            raise HTTPException(
                status_code=400,
                detail=f"Kakao geocoding network error for '{query}': {e}",
            )

        if not r.ok:
            raise HTTPException(
                status_code=400,
                detail=f"Kakao HTTP {r.status_code} for '{query}': {r.text}",
            )

        try:
            return r.json()
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Kakao JSON parse error for '{query}': {e}, body={r.text}",
            )

    # ----- 5) Build candidate queries -----
    candidates = []
    # Full line first
    candidates.append(line)

    # Strip '아파트'
    if "아파트" in line:
        no_apt = line.replace("아파트", "").strip()
        if no_apt and no_apt != line:
            candidates.append(no_apt)

    # Last 2 tokens (e.g. '래미안 베네루체')
    tokens = line.split()
    if len(tokens) >= 2:
        last2 = " ".join(tokens[-2:])
        if last2 not in candidates:
            candidates.append(last2)

    # Optionally you can also try just the last token
    if len(tokens) >= 1:
        last1 = tokens[-1]
        if last1 not in candidates:
            candidates.append(last1)

    docs = None
    used_query = None

    # ----- 6) Try candidates in order -----
    for q in candidates:
        if not q:
            continue
        data = kakao_search(q)
        d = data.get("documents", [])
        if d:
            docs = d
            used_query = q
            break

    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"Could not geocode location via Kakao: '{line}'",
        )

    doc = docs[0]
    try:
        lon = float(doc["x"])
        lat = float(doc["y"])
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid Kakao geocoding result for '{line}': {e}",
        )

    name = doc.get("place_name") or used_query or line
    road_addr = doc.get("road_address_name")
    addr = doc.get("address_name")
    label_parts = [p for p in [name, road_addr, addr] if p]
    label = " · ".join(label_parts) if label_parts else line

    GEOCODE_CACHE[line] = (lat, lon, label)
    return lat, lon, label



# -----------------------------------------------------------------------------
# OSRM route & step fetching
# -----------------------------------------------------------------------------

def fetch_osrm_steps(route_coords: List[Tuple[float, float]]):
    """
    Call OSRM public server to get turn-by-turn steps and their geometry.

    route_coords: list of (lat, lon) in order of the path.
    Returns:
      osrm_steps: list of dicts
          { instruction, distance_m, duration_s, polyline: [[lat,lon], ...] }
    """
    if len(route_coords) < 2:
        return []

    # OSRM expects "lon,lat" order
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in route_coords)

    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        # If OSRM fails, we just return empty steps; the rest of the app still works.
        print("OSRM error:", e)
        return []

    routes = data.get("routes")
    if not routes:
        return []

    route = routes[0]
    legs = route.get("legs", [])

    osrm_steps = []
    # enumerate 로 leg_idx 를 정의해서 각 step 에 leg_index 를 넣어줌
    for leg_idx, leg in enumerate(legs):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            inst = maneuver.get("instruction") or step.get("name") or "Continue"
            distance_m = float(step.get("distance", 0.0))
            duration_s = float(step.get("duration", 0.0))
            geom = step.get("geometry", {})
            coords = geom.get("coordinates", [])
            polyline = [[lat, lon] for lon, lat in coords]  # flip to [lat,lon]

            osrm_steps.append(
                {
                    "instruction": inst,
                    "distance_m": distance_m,
                    "duration_s": duration_s,
                    "polyline": polyline,
                    "leg_index": leg_idx,
                }
            )

    return osrm_steps


# -----------------------------------------------------------------------------
# Leg summary builder (approx distances/time between consecutive stops)
# -----------------------------------------------------------------------------

def build_leg_summaries(route_indices: List[int], locations: List[Location], labels: List[str]):
    legs = []
    for i in range(len(route_indices) - 1):
        a_idx = route_indices[i]
        b_idx = route_indices[i + 1]
        a = (locations[a_idx].lat, locations[a_idx].lon)
        b = (locations[b_idx].lat, locations[b_idx].lon)
        dist_km = haversine_km(a, b)
        time_hours = dist_km / 30.0
        legs.append(
            {
                "from_index": a_idx,
                "to_index": b_idx,
                "from_name": labels[a_idx],
                "to_name": labels[b_idx],
                "distance_km": dist_km,
                "time_min": time_hours * 60.0,
            }
        )
    return legs


# -----------------------------------------------------------------------------
# API endpoint: optimize-by-name
# -----------------------------------------------------------------------------

@app.post("/optimize-by-name")
def optimize_by_name(req: OptimizeByNameRequest):
    if len(req.lines) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 locations.")

    # Geocode all lines
    locations: List[Location] = []
    labels: List[str] = []
    for raw in req.lines:
        lat, lon, label = geocode_line(raw)
        locations.append(Location(lat=lat, lon=lon))
        labels.append(label)

    # Run route optimizer
    result = optimize_core(locations, req.start_index, req.end_index)
    route_indices = result["route_indices"]
    total_time_seconds = result["total_time_seconds"]
    total_time_hours = result["total_time_hours"]
    solver_mode = result["solver_mode"]
    solver_cost = result["solver_cost"]

    # Build stops list for frontend
    stops_out = []
    for loc, label in zip(locations, labels):
        stops_out.append(
            {
                "label": label,
                "lat": loc.lat,
                "lon": loc.lon,
            }
        )

    # Build approximate leg summaries (using haversine distances)
    legs = build_leg_summaries(route_indices, locations, labels)

    # For OSRM steps, we feed coordinates in route order
    route_coords = [(locations[i].lat, locations[i].lon) for i in route_indices]
    osrm_steps = fetch_osrm_steps(route_coords)

    return {
        "route_indices": route_indices,
        "total_time_seconds": total_time_seconds,
        "total_time_hours": total_time_hours,
        "solver_mode": solver_mode,
        "solver_cost": solver_cost,
        "stops": stops_out,
        "legs": legs,
        "osrm_steps": osrm_steps,
    }


# -----------------------------------------------------------------------------
# Small root endpoint just to check server
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Multi-stop route optimizer backend"}
