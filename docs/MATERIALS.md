# Recommended Materials

Material recommendations per component, optimised for low cost, availability, and adequate strength at the loads this rocket sees.

## Rocket

| Component | Material | Rationale |
|-----------|----------|-----------|
| **Nose cone** | 3D-printed PLA or PETG | Low stress; PLA is cheapest, PETG for outdoor heat tolerance |
| **Body tube** | Fiberglass or thin-wall carbon-fibre tube (75 mm OD) | Survives motor mount thermal load and bending at fin root better than PLA |
| **Fins (tail)** | Glass-filled nylon (SLS/MJF) or thin CFRP sheet | Needs high stiffness under aero load; folding hinge root is the critical joint |
| **Canards** | Glass-filled nylon or PETG | Must handle servo torque and aero side-load; glass nylon preferred |
| **Motor mount** | Aluminium retainer ring + phenolic tube | Standard model-rocket practice; phenolic insulates the body tube from exhaust heat |
| **Rail buttons** | Nylon or Delrin (acetal) | Low friction, self-lubricating on aluminium rail |
| **Fasteners** | M2/M3 stainless-steel machine screws + nylon lock nuts | Vibration-resistant; stainless avoids galvanic issues with aluminium |

## Launcher

| Component | Material | Rationale |
|-----------|----------|-----------|
| **Rails** | 2020 aluminium V-slot extrusion (two parallel, ~1 m each) | Inexpensive, straight, standard hobby extrusion; V-slot guides the rail buttons |
| **Base plate** | 6 mm aluminium plate or 3D-printed PETG bracket | Needs rigidity to resist launch-force reaction; aluminium preferred outdoors |
| **Igniter mount** | 3D-printed PETG bracket + steel retainer clip | Heat-exposed but brief; PETG adequate if insulated from nozzle blast |
| **Electronics enclosure** | IP65 plastic junction box or 3D-printed PETG | Protects ESP32, battery, and servo driver from weather |

## Cost Estimates (Approximate)

| Item | Estimated Cost |
|------|---------------|
| Fiberglass body tube (75 mm OD × 600 mm) | $8–15 |
| Glass-filled nylon fins (SLS print, set of 4) | $15–25 |
| PETG canards (FDM, set of 4) | $2–5 |
| PLA nose cone | $1–3 |
| Aluminium motor retainer | $5–8 |
| 2020 extrusion rails (2 × 1 m) | $8–12 |
| Aluminium base plate (200 × 150 × 6 mm) | $5–10 |
| Nylon rail buttons (pair) | $1–2 |
| **Total materials (rocket + launcher)** | **~$45–80** |

Electronics, motor, and servo costs are separate and documented in the main README.
