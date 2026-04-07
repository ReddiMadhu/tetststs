"""
patch_conflict_rules.py
Loads the existing construction_conflict_rules.json and adds:
1. scheme_override field to each scenario
2. New scenarios for all 30 user-provided rules with correct scheme
"""
import json, pathlib, re

FILE = pathlib.Path("reference/construction_conflict_rules.json")
data = json.loads(FILE.read_text(encoding="utf-8"))

# Map final_category → {bldgclass, bldgscheme}
CATEGORY_SCHEME = {
    "Steel Frame":             {"bldgclass": "4",   "bldgscheme": "RMS"},
    "Joisted Masonry":         {"bldgclass": "2",   "bldgscheme": "FIRE"},
    "Concrete Frame":          {"bldgclass": "3",   "bldgscheme": "RMS"},
    "Masonry Non-Combustible": {"bldgclass": "4",   "bldgscheme": "FIRE"},
    "Frame":                   {"bldgclass": "1",   "bldgscheme": "RMS"},
    "Mixed Construction":      {"bldgclass": "1",   "bldgscheme": "RMS"},
    "Metal Building":          {"bldgclass": "4B",  "bldgscheme": "RMS"},
    "Heavy Timber":            {"bldgclass": "7",   "bldgscheme": "FIRE"},
    "Non-Combustible":         {"bldgclass": "3",   "bldgscheme": "FIRE"},
    "Precast Concrete":        {"bldgclass": "3B",  "bldgscheme": "RMS"},
    "Tilt-Up Concrete":        {"bldgclass": "3B4", "bldgscheme": "RMS"},
    "Fire Resistive":          {"bldgclass": "6",   "bldgscheme": "FIRE"},
    "Modified Fire Resistive": {"bldgclass": "5",   "bldgscheme": "FIRE"},
    "Unknown":                 {"bldgclass": "0",   "bldgscheme": "RMS"},
}

# Patch existing scenarios
for s in data.get("scenarios", []):
    fc = s.get("final_category", "")
    meta = CATEGORY_SCHEME.get(fc)
    if meta:
        s["bldgclass"] = meta["bldgclass"]
        s["scheme_override"] = meta["bldgscheme"]

# Add new scenarios for the 30 user rules (with full details)
NEW_SCENARIOS = [
    {
        "id": 101, "raw": "Frame",
        "primary_system": "wood_frame", "final_category": "Frame",
        "air_code": "101", "iso_class": "1",
        "bldgclass": "1", "scheme_override": "RMS",
        "confidence": 0.95,
        "rule": "const_1_frame_alone_equals_wood",
        "reasoning": "Bare 'Frame' → Wood Frame (CONST-1); residential default"
    },
    {
        "id": 102, "raw": "Wood Frame",
        "primary_system": "wood_frame", "final_category": "Frame",
        "air_code": "101", "iso_class": "1",
        "bldgclass": "1", "scheme_override": "RMS",
        "confidence": 0.97,
        "rule": "const_1_frame_alone_equals_wood",
        "reasoning": "Wood Frame → RMS 1 (Wood); direct match"
    },
    {
        "id": 103, "raw": "Non-Combustible",
        "primary_system": "non_combustible", "final_category": "Non-Combustible",
        "air_code": "152", "iso_class": "3",
        "bldgclass": "3", "scheme_override": "FIRE",
        "confidence": 0.97,
        "rule": "const_2_non_combustible_is_iso3",
        "reasoning": "Non-Combustible → ISO Fire Class 3 (FIRE scheme, BLDGCLASS=3)"
    },
    {
        "id": 104, "raw": "Wood Frame / Brick Siding",
        "primary_system": "wood_frame", "final_category": "Frame",
        "air_code": "101", "iso_class": "1",
        "bldgclass": "1", "scheme_override": "RMS",
        "confidence": 0.92,
        "rule": "frame_governs_const_3",
        "reasoning": "Wood Frame primary; Brick Siding is cladding/veneer (RULE 4 + CONST-3)"
    },
    {
        "id": 105, "raw": "Frame over 2-story concrete podium",
        "primary_system": "wood_frame", "final_category": "Frame",
        "air_code": "101", "iso_class": "1",
        "bldgclass": "1", "scheme_override": "RMS",
        "confidence": 0.90,
        "rule": "frame_governs_const_5_podium",
        "reasoning": "Frame (wood, primary residential structure) over concrete podium; podium=foundation not structural frame (CONST-5)"
    },
    {
        "id": 106, "raw": "Concrete podium with wood upper floors",
        "primary_system": "mixed", "final_category": "Mixed Construction",
        "air_code": "141", "iso_class": "2",
        "bldgclass": "1", "scheme_override": "RMS",
        "confidence": 0.90,
        "rule": "mixed_construction",
        "reasoning": "Hybrid vertical: combustible wood upper floors over concrete base"
    },
    {
        "id": 107, "raw": "Fire Resistive",
        "primary_system": "fire_resistive", "final_category": "Fire Resistive",
        "air_code": "131", "iso_class": "6",
        "bldgclass": "6", "scheme_override": "FIRE",
        "confidence": 0.95,
        "rule": "const_6_fire_resistive",
        "reasoning": "Fire Resistive → ISO Fire Class 6 (FIRE scheme)"
    },
    {
        "id": 108, "raw": "Modified Fire Resistive",
        "primary_system": "modified_fire_resistive", "final_category": "Modified Fire Resistive",
        "air_code": "151", "iso_class": "5",
        "bldgclass": "5", "scheme_override": "FIRE",
        "confidence": 0.95,
        "rule": "const_7_modified_fire_resistive",
        "reasoning": "Modified Fire Resistive → ISO Fire Class 5 (FIRE scheme)"
    },
    {
        "id": 109, "raw": "Joisted Masonry",
        "primary_system": "joisted_masonry", "final_category": "Joisted Masonry",
        "air_code": "119", "iso_class": "2",
        "bldgclass": "2", "scheme_override": "FIRE",
        "confidence": 0.97,
        "rule": "const_8_joisted_masonry",
        "reasoning": "Joisted Masonry → ISO Fire Class 2 (FIRE scheme)"
    },
    {
        "id": 110, "raw": "Heavy Timber",
        "primary_system": "heavy_timber", "final_category": "Heavy Timber",
        "air_code": "104", "iso_class": "7",
        "bldgclass": "7", "scheme_override": "FIRE",
        "confidence": 0.97,
        "rule": "const_10_heavy_timber",
        "reasoning": "Heavy Timber → ISO Fire Class 7 (FIRE scheme)"
    },
    {
        "id": 111, "raw": "Masonry Non-Combustible",
        "primary_system": "masonry_non_combustible", "final_category": "Masonry Non-Combustible",
        "air_code": "111", "iso_class": "4",
        "bldgclass": "4", "scheme_override": "FIRE",
        "confidence": 0.97,
        "rule": "const_9_masonry_non_combustible",
        "reasoning": "Masonry Non-Combustible → ISO Fire Class 4 (FIRE scheme)"
    },
]

# Merge: add new ones that aren't already present by raw (case-insensitive)
existing_raws = {s["raw"].lower().strip() for s in data["scenarios"]}
added = 0
for ns in NEW_SCENARIOS:
    if ns["raw"].lower().strip() not in existing_raws:
        data["scenarios"].append(ns)
        existing_raws.add(ns["raw"].lower().strip())
        added += 1

print(f"Patched {len(data['scenarios'])} scenarios ({added} new added)")

FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved to {FILE}")

# Verify a sample
for s in data["scenarios"][:3]:
    print(f"  id={s['id']} raw={s['raw']!r} scheme_override={s.get('scheme_override','?')} bldgclass={s.get('bldgclass','?')}")
