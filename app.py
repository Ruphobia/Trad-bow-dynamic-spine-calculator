#!/usr/bin/python3
"""
ARCHERY BARE-SHAFT BRACKET TOOL (CLI)

This program brackets arrow spine with two point weights (75 gr and 300 gr),
collects component weights and purpose, then computes build options across
accepted insert weights. It now shows ALL setups — those BELOW the target FOC,
those INSIDE the target FOC window, and those ABOVE the target FOC — and sorts
them by HIGHEST FOC FIRST.

Run: python3 bracket_tool.py
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import csv
import datetime
import sys


# ----------------------------- Data Models ----------------------------- #

@dataclass
class PurposeSpec:
    """Holds purpose-specific target windows.

    Parameters:
        name: str -> purpose label.
        total_tip_min: int -> minimum total tip weight in grains.
        total_tip_max: int -> maximum total tip weight in grains.
        finished_min: int -> recommended minimum finished arrow mass in grains.
        finished_max: int -> recommended maximum finished arrow mass in grains.
        foc_min_percent: float -> minimum FOC percent.
        foc_max_percent: float -> maximum FOC percent.

    Returns:
        None
    """
    name: str
    total_tip_min: int
    total_tip_max: int
    finished_min: int
    finished_max: int
    foc_min_percent: float
    foc_max_percent: float


@dataclass
class SpineBracketResult:
    """Stores bracket results for a given spine at a given length.

    Parameters:
        spine: int -> spine value like 300, 350.
        length_in: float -> carbon length in inches.
        res_75: Optional[str] -> 'stiff' | 'weak' | 'neutral' | None.
        res_300: Optional[str] -> 'stiff' | 'weak' | 'neutral' | None.
        verdict: str -> verdict text for the pair.

    Returns:
        None
    """
    spine: int
    length_in: float
    res_75: Optional[str]
    res_300: Optional[str]
    verdict: str


@dataclass
class BuildRow:
    """Represents a computed build row for display and CSV export.

    Parameters:
        spine: int -> spine value.
        window_side: str -> stronger side | weaker side | neutral | too stiff | too weak | skipped.
        shaft_mass_gr: float -> shaft mass in grains.
        point_gr: int -> point weight in grains.
        insert_gr: int -> insert weight in grains.
        total_tip_gr: int -> total tip weight in grains.
        nock_gr: float -> nock weight in grains.
        fletch_gr: float -> fletching weight in grains.
        total_mass_gr: float -> finished arrow mass in grains.
        calc_foc_percent: float -> calculated FOC percent.
        target_foc_str: str -> target FOC range text.
        foc_status: str -> 'Below', 'In', 'Above' target FOC window.

    Returns:
        None
    """
    spine: int
    window_side: str
    shaft_mass_gr: float
    point_gr: int
    insert_gr: int
    total_tip_gr: int
    nock_gr: float
    fletch_gr: float
    total_mass_gr: float
    calc_foc_percent: float
    target_foc_str: str
    foc_status: str


# ----------------------------- Constants ----------------------------- #

SPINES: List[int] = [300, 350, 400, 500, 600]

# Default GPI per your data (uncut length 33 in); press Enter to accept in UI.
SHAFT_GPI_DEFAULTS: Dict[int, float] = {
    300: 8.452,  # 278.9 gr at 33"
    350: 7.715,  # 254.6 gr at 33"
    400: 7.282,  # 240.3 gr at 33"
    500: 5.676,  # 187.3 gr at 33"
    600: 5.079,  # 167.6 gr at 33"
}

# Acceptable insert weights menu, deduped and sorted.
# Adjust this list to match the hardware you actually own / can buy.
ACCEPTABLE_INSERT_WEIGHTS: List[int] = [12, 15, 20, 25, 30, 50, 75, 100]

# Purposes table.
PURPOSES: List[PurposeSpec] = [
    PurposeSpec("Target / practice", 75, 150, 350, 550, 8.0, 15.0),
    PurposeSpec("Small game (squirrels, rabbits, birds)", 100, 150, 450, 500, 8.0, 12.0),
    PurposeSpec("Coyote / fox / turkey (medium, standard FOC)", 100, 125, 400, 500, 10.0, 15.0),
    PurposeSpec("Coyote / fox / turkey (medium, high FOC)", 150, 175, 500, 550, 12.0, 18.0),
    PurposeSpec("Deer / antelope (standard FOC)", 100, 150, 450, 550, 12.0, 15.0),
    PurposeSpec("Deer / antelope (high FOC)", 150, 200, 525, 650, 15.0, 18.0),
    PurposeSpec("Black bear / mountain goat / larger deer", 175, 225, 550, 625, 15.0, 20.0),
    PurposeSpec("Elk / caribou", 200, 250, 600, 675, 15.0, 20.0),
    PurposeSpec("Moose / bison / large boar / exotics", 250, 300, 650, 750, 18.0, 25.0),
    PurposeSpec("African big game (Cape buffalo, etc.)", 300, 350, 750, 900, 18.0, 25.0),
]

SHAFT_LENGTH_OPTIONS: List[float] = [
    32.0, 31.5, 31.0, 30.5, 30.0, 29.5, 29.0, 28.5, 28.0, 27.0, 26.0, 25.0
]


# ----------------------------- Helpers ----------------------------- #

def prompt_int(prompt: str, min_val: Optional[int] = None, max_val: Optional[int] = None, default: Optional[int] = None) -> int:
    """Prompt for an integer.

    Parameters:
        prompt: str -> text to show.
        min_val: Optional[int] -> minimum allowed or None.
        max_val: Optional[int] -> maximum allowed or None.
        default: Optional[int] -> default returned if user presses Enter.

    Returns:
        int -> the chosen integer.
    """
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            v = int(s)
            if min_val is not None and v < min_val:
                print(f"Enter a number >= {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"Enter a number <= {max_val}.")
                continue
            return v
        except ValueError:
            print("Enter a whole number.")


def prompt_float(prompt: str, min_val: Optional[float] = None, max_val: Optional[float] = None, default: Optional[float] = None) -> float:
    """Prompt for a float.

    Parameters:
        prompt: str -> text to show.
        min_val: Optional[float] -> minimum allowed or None.
        max_val: Optional[float] -> maximum allowed or None.
        default: Optional[float] -> default returned if user presses Enter.

    Returns:
        float -> the chosen float.
    """
    while True:
        s = input(prompt).strip()
        if s == "" and default is not None:
            return default
        try:
            v = float(s)
            if min_val is not None and v < min_val:
                print(f"Enter a value >= {min_val}.")
                continue
            if max_val is not None and v > max_val:
                print(f"Enter a value <= {max_val}.")
                continue
            return v
        except ValueError:
            print("Enter a number.")


def map_nock_to_result(handed: str, val: int) -> Optional[str]:
    """Map menu selection to 'stiff' | 'weak' | 'neutral' or None for skip.

    Parameters:
        handed: str -> 'Right' or 'Left'.
        val: int -> 1 left, 2 right, 3 center, 4 skip.

    Returns:
        Optional[str] -> mapped result or None if skipped.
    """
    if val == 4:
        return None
    if val == 3:
        return "neutral"
    # 1 = Left, 2 = Right
    if handed == "Right":
        return "weak" if val == 1 else "stiff"
    else:
        return "stiff" if val == 1 else "weak"


def classify_verdict(res_75: Optional[str], res_300: Optional[str]) -> str:
    """Classify pair of 75 and 300 results into a verdict string.

    Parameters:
        res_75: Optional[str] -> 'stiff' | 'weak' | 'neutral' | None.
        res_300: Optional[str] -> 'stiff' | 'weak' | 'neutral' | None.

    Returns:
        str -> verdict text.
    """
    if res_75 is None or res_300 is None:
        return "skipped"
    if res_75 == "stiff" and res_300 == "weak":
        return "in window"
    if res_75 == "stiff" and res_300 == "neutral":
        return "in window (stronger side)"
    if res_75 == "neutral" and res_300 == "weak":
        return "in window (weaker side)"
    if res_75 == "stiff" and res_300 == "stiff":
        return "too stiff"
    if res_75 == "weak" and res_300 == "weak":
        return "too weak"
    if res_75 == "neutral" and res_300 == "neutral":
        return "in window"
    if res_75 == "neutral" and res_300 == "stiff":
        return "too stiff"
    if res_75 == "weak" and res_300 == "neutral":
        return "too weak"
    return "skipped"


def compute_foc_percent(
    shaft_len_in: float,
    shaft_mass_gr: float,
    total_tip_gr: int,
    nock_gr: float,
    fletch_gr: float,
    fletch_pos_in_from_nock: float = 1.0
) -> float:
    """Compute FOC percent using a simple 1D center-of-mass model.

    Parameters:
        shaft_len_in: float -> carbon length in inches.
        shaft_mass_gr: float -> mass of carbon tube in grains.
        total_tip_gr: int -> point plus insert in grains.
        nock_gr: float -> nock weight in grains.
        fletch_gr: float -> total fletching weight in grains.
        fletch_pos_in_from_nock: float -> center of fletching position from nock in inches.

    Returns:
        float -> FOC percent.
    """
    L = shaft_len_in
    pos_shaft = L / 2.0
    pos_tip = L
    pos_nock = 0.0
    pos_fletch = max(0.0, min(L, fletch_pos_in_from_nock))

    total_mass = shaft_mass_gr + total_tip_gr + nock_gr + fletch_gr
    if total_mass <= 0.0:
        return 0.0

    moment = (
        shaft_mass_gr * pos_shaft
        + total_tip_gr * pos_tip
        + nock_gr * pos_nock
        + fletch_gr * pos_fletch
    )
    balance_point = moment / total_mass
    foc = ((balance_point - (L / 2.0)) / L) * 100.0
    return foc


def verdict_window_side(verdict: str) -> str:
    """Extract window side string for display.

    Parameters:
        verdict: str -> verdict text.

    Returns:
        str -> side text.
    """
    if "stronger" in verdict:
        return "Stronger side"
    if "weaker" in verdict:
        return "Weaker side"
    if verdict == "in window":
        return "Neutral"
    if verdict == "too stiff":
        return "Too stiff"
    if verdict == "too weak":
        return "Too weak"
    return "Skipped"


def export_csv(path: str, rows: List[BuildRow], header: Dict[str, str]) -> None:
    """Export the final table to CSV.

    Parameters:
        path: str -> file path to write.
        rows: List[BuildRow] -> computed rows to export.
        header: Dict[str, str] -> metadata header written as comments.

    Returns:
        None
    """
    with open(path, "w", newline="") as f:
        # Header lines as comments
        for k, v in header.items():
            f.write(f"# {k}: {v}\n")
        writer = csv.writer(f)
        writer.writerow([
            "Spine", "Window Side", "Shaft Mass (gr)", "Point (gr)", "Insert (gr)",
            "Total Tip (gr)", "Nock (gr)", "Fletch (gr)", "Total Mass (gr)",
            "Calc FOC (%)", "Target FOC", "FOC Status"
        ])
        for r in rows:
            writer.writerow([
                r.spine,
                r.window_side,
                f"{r.shaft_mass_gr:.1f}",
                r.point_gr,
                r.insert_gr,
                r.total_tip_gr,
                f"{r.nock_gr:.1f}",
                f"{r.fletch_gr:.1f}",
                f"{r.total_mass_gr:.1f}",
                f"{r.calc_foc_percent:.1f}",
                r.target_foc_str,
                r.foc_status
            ])


# ----------------------------- Main Flow ----------------------------- #

def main() -> None:
    """Entry point for the CLI program.

    Parameters:
        None

    Returns:
        None
    """
    print("ARCHERY BARE-SHAFT BRACKET TOOL")
    print("--------------------------------")
    print("1) Begin a new session")
    print("2) Exit")
    sel = prompt_int("Select: ", 1, 2)
    if sel == 2:
        sys.exit(0)

    # Handedness
    print("\nHandedness:")
    print("1) Right-handed")
    print("2) Left-handed")
    handed_sel = prompt_int("Select: ", 1, 2)
    handed = "Right" if handed_sel == 1 else "Left"

    # Shaft length
    print("\nPick test shaft length (carbon), enter the number:")
    for i, L in enumerate(SHAFT_LENGTH_OPTIONS, 1):
        print(f" {i}) {L:.1f}\"")
    idx = prompt_int("Select (1..12): ", 1, len(SHAFT_LENGTH_OPTIONS))
    shaft_len_in = SHAFT_LENGTH_OPTIONS[idx - 1]

    # Optional override of default GPI per spine
    print("\nShaft GPI per spine (press Enter to accept defaults):")
    gpi: Dict[int, float] = {}
    for sp in SPINES:
        gpi[sp] = prompt_float(
            f"  GPI for spine {sp} [default {SHAFT_GPI_DEFAULTS[sp]}]: ",
            min_val=3.0, max_val=30.0, default=SHAFT_GPI_DEFAULTS[sp]
        )

    print("\nLegend:")
    if handed == "Right":
        print("- Right-handed shooter: Nock right = stiff, Nock left = weak, Center = neutral")
    else:
        print("- Left-handed shooter:  Nock right = weak,  Nock left = stiff, Center = neutral")

    print("\n-------------------------------------------------------")
    print(f"Enter bracket results for each spine at {shaft_len_in:.1f}\" carbon")
    print("75 gr point then 300 gr point; choose 4 to skip any spine")
    print("-------------------------------------------------------")

    bracket_results: List[SpineBracketResult] = []
    for sp in SPINES:
        print(f"\nTesting spine: {sp}")
        s75 = prompt_int("75 gr point (1=Left,2=Right,3=Center,4=Skip): ", 1, 4)
        res_75 = map_nock_to_result(handed, s75)
        res_300 = None
        if res_75 is not None:
            s300 = prompt_int("300 gr point (1=Left,2=Right,3=Center,4=Skip): ", 1, 4)
            res_300 = map_nock_to_result(handed, s300)
        verdict = classify_verdict(res_75, res_300)

        # Echo result
        def rtxt(x: Optional[str]) -> str:
            """Format result for printing.

            Parameters:
                x: Optional[str] -> 'stiff' | 'weak' | 'neutral' | None.

            Returns:
                str -> formatted text token.
            """
            return "skipped" if x is None else x

        print(f"Result ({sp} @ {shaft_len_in:.1f}\"):")
        if res_75 is not None:
            print(f" - 75 gr: {rtxt(res_75)}")
        else:
            print(" - 75 gr: skipped")
        if res_300 is not None:
            print(f" - 300 gr: {rtxt(res_300)}")
        else:
            print(" - 300 gr: skipped")
        print(f"Conclusion: {verdict}.")

        bracket_results.append(SpineBracketResult(
            spine=sp,
            length_in=shaft_len_in,
            res_75=res_75,
            res_300=res_300,
            verdict=verdict
        ))

    # Component weights and purpose
    print("\n-------------------------------------------------------")
    print("Enter arrow component weights")
    print("-------------------------------------------------------")
    nock_gr = prompt_float("Enter nock weight in grains [default 9.1]: ", min_val=0.0, max_val=100.0, default=9.1)
    point_gr = prompt_int("Enter target tip weight (your broadhead/field point) [default 100]: ", min_val=50, max_val=500, default=100)
    fletch_gr = prompt_float("Enter total fletching weight in grains: ", min_val=0.0, max_val=200.0, default=24.0)

    print("\nSelect arrow purpose:")
    for i, p in enumerate(PURPOSES, 1):
        print(f"{i}) {p.name}")
    psel = prompt_int("Select: ", 1, len(PURPOSES))
    purpose = PURPOSES[psel - 1]

    # Build ALL candidate rows (no FOC filter) within purpose's total tip range.
    # Only consider spines that aren't 'too stiff', 'too weak', or 'skipped'.
    rows: List[BuildRow] = []
    for br in bracket_results:
        v = br.verdict
        if v in ("too stiff", "too weak", "skipped"):
            continue

        wside = verdict_window_side(v)

        for ins in ACCEPTABLE_INSERT_WEIGHTS:
            total_tip = point_gr + ins
            # Keep only total tip combos that are within the purpose total-tip window.
            if total_tip < purpose.total_tip_min or total_tip > purpose.total_tip_max:
                continue

            shaft_mass = gpi[br.spine] * shaft_len_in
            total_mass = shaft_mass + total_tip + nock_gr + fletch_gr
            foc = compute_foc_percent(
                shaft_len_in=shaft_len_in,
                shaft_mass_gr=shaft_mass,
                total_tip_gr=total_tip,
                nock_gr=nock_gr,
                fletch_gr=fletch_gr,
                fletch_pos_in_from_nock=1.0
            )

            # Determine FOC status relative to target window (Below / In / Above)
            if foc < purpose.foc_min_percent:
                status = "Below"
            elif foc > purpose.foc_max_percent:
                status = "Above"
            else:
                status = "In"

            rows.append(BuildRow(
                spine=br.spine,
                window_side=wside,
                shaft_mass_gr=shaft_mass,
                point_gr=point_gr,
                insert_gr=ins,
                total_tip_gr=total_tip,
                nock_gr=nock_gr,
                fletch_gr=fletch_gr,
                total_mass_gr=total_mass,
                calc_foc_percent=foc,
                target_foc_str=f"{int(purpose.foc_min_percent)}–{int(purpose.foc_max_percent)}%",
                foc_status=status
            ))

    # Sort: highest FOC first, then lighter total mass, then stiffer spine as tiebreaker.
    rows.sort(key=lambda r: (-r.calc_foc_percent, r.total_mass_gr, r.spine))

    # Print results
    print("\n-------------------------------------------------------")
    print(f"RECOMMENDED BUILDS @ {shaft_len_in:.1f}\" — Sorted by Highest FOC (Shows Below / In / Above)")
    print(f"Purpose: {purpose.name} | Target Total Tip {purpose.total_tip_min}–{purpose.total_tip_max} gr | Target FOC {int(purpose.foc_min_percent)}–{int(purpose.foc_max_percent)}%")
    print("-------------------------------------------------------")

    if not rows:
        print("No builds matched the total tip window. Adjust insert list or point weight.")
    else:
        header = (
            "Spine  Window Side            Shaft Mass  Point  Insert  Total Tip  Nock  Fletch  Total Mass  Calc FOC  Target FOC  FOC Status"
        )
        print(header)
        print("-" * len(header))
        for r in rows:
            print(
                f"{r.spine:<6}{r.window_side:<23}"
                f"{r.shaft_mass_gr:>8.1f}    "
                f"{r.point_gr:>5}  "
                f"{r.insert_gr:>6}    "
                f"{r.total_tip_gr:>9}  "
                f"{r.nock_gr:>4.1f}  "
                f"{r.fletch_gr:>6.1f}    "
                f"{r.total_mass_gr:>10.1f}    "
                f"{r.calc_foc_percent:>7.1f}%    "
                f"{r.target_foc_str:>9}  "
                f"{r.foc_status}"
            )

    # Session summary with option to export
    print("\nNext:")
    print("1) Start a new session with a different shaft length")
    print("2) Export summary to CSV")
    print("3) Exit")
    nsel = prompt_int("Select: ", 1, 3)
    if nsel == 1:
        print()
        main()
        return
    if nsel == 2:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = f"./bracket_{ts}.csv"
        meta = {
            "Handedness": handed,
            "Length": f"{shaft_len_in:.1f} in",
            "Purpose": purpose.name,
            "Target Total Tip": f"{purpose.total_tip_min}-{purpose.total_tip_max} gr",
            "Target FOC": f"{int(purpose.foc_min_percent)}-{int(purpose.foc_max_percent)}%",
            "Point": f"{point_gr} gr",
            "Nock": f"{nock_gr:.1f} gr",
            "Fletch": f"{fletch_gr:.1f} gr",
            "Accepted Inserts": ", ".join(str(w) for w in ACCEPTABLE_INSERT_WEIGHTS)
        }
        export_csv(path, rows, meta)
        print(f"Saved: {path}")
    print("Done.")


# ----------------------------- Entrypoint ----------------------------- #

if __name__ == "__main__":
    main()
