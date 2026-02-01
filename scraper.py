import re
import pandas as pd
import argparse
import os

POWERS = ["Austria", "England", "France", "Germany", "Italy", "Russia", "Turkey"]

round_re = re.compile(r"^\[Round\s+(\d+)\]\s+Phase=(\w+)", re.IGNORECASE)
proposal_line_re = re.compile(r"^\s*([A-Za-z]+)\s*->\s*(.*)$")
deal_pair_re = re.compile(r"^\s*([A-Za-z]+)\s*<->\s*([A-Za-z]+)\b")

# Agent roster parsing
roster_start_re = re.compile(r"^\[mixed_tom\]\s*Agent roster:", re.IGNORECASE)
roster_line_re = re.compile(r"^\s*([A-Za-z]+)\s*:\s*.*\(\s*tom_depth\s*=\s*(\d+)\s*\)", re.IGNORECASE)

def norm_list(s: str):
    s = s.strip()
    if s.lower() in {"(none)", "none", ""}:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def parse_roster_tom_depth(lines):
    """
    Reads ToM depth per power from the top-of-file roster.
    Returns dict: {power: tom_depth_int}
    """
    roster = {}
    in_roster = False

    for line in lines:
        line = line.rstrip("\n")

        if roster_start_re.match(line):
            in_roster = True
            continue

        if in_roster:
            # roster block ends when we hit next [mixed_tom] line or first [Round ...]
            if line.startswith("[mixed_tom]") or line.startswith("[Round"):
                break

            m = roster_line_re.match(line)
            if m:
                p = m.group(1)
                if p in POWERS:
                    roster[p] = int(m.group(2))

    return roster

def process_log(log_path: str, out_csv: str):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    roster_tom = parse_roster_tom_depth(lines)

    # hard fail if roster is missing/partial (because you said it's important)
    missing = [p for p in POWERS if p not in roster_tom]
    if missing:
        raise ValueError(
            f"Missing tom_depth in Agent roster for: {missing}. "
            f"Expected lines like: 'Austria: ... (tom_depth=2)'."
        )

    rows = []
    i = 0

    while i < len(lines):
        line = lines[i].rstrip("\n")

        m = round_re.match(line)
        if m:
            current_round = int(m.group(1))
            current_phase = m.group(2).upper()

            proposals = {p: [] for p in POWERS}
            accepted_with = {p: set() for p in POWERS}

            i += 1
            section = None
            while i < len(lines):
                l = lines[i].rstrip("\n")
                if round_re.match(l):
                    i -= 1
                    break

                if l.strip() == "Proposals:":
                    section = "proposals"
                elif l.strip() == "Active deals:":
                    section = "deals"
                elif re.match(r"^\s*Deal stats:", l):
                    section = "other"
                elif re.match(r"^\s*Orders:", l):
                    section = "other"

                if section == "proposals":
                    mm = proposal_line_re.match(l)
                    if mm:
                        sender = mm.group(1)
                        if sender in proposals:
                            proposals[sender] = norm_list(mm.group(2))

                if section == "deals":
                    mm = deal_pair_re.match(l)
                    if mm:
                        a, b = mm.group(1), mm.group(2)
                        if a in accepted_with and b in accepted_with:
                            accepted_with[a].add(b)
                            accepted_with[b].add(a)

                i += 1

            received_from = {p: [] for p in POWERS}
            for sender, targets in proposals.items():
                for t in targets:
                    if t in received_from:
                        received_from[t].append(sender)

            for p in POWERS:
                sent_to = proposals[p]
                acc = sorted(accepted_with[p])
                rows.append({
                    "round": current_round,
                    "phase": current_phase,
                    "power": p,
                    "tom_depth": roster_tom[p],  # <-- the important bit

                    "proposals_sent_to": ";".join(sent_to) if sent_to else "",
                    "accepted_deals_with": ";".join(acc) if acc else "",
                    "sent_count": len(sent_to),
                    "accepted_count": len(acc),
                    "rejected_count": max(0, len(sent_to) - len(acc)),
                    "proposals_received_from": ";".join(received_from[p]) if received_from[p] else "",
                    "received_count": len(received_from[p]),
                })

        i += 1

    df = pd.DataFrame(rows).sort_values(["round", "phase", "power"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse mixed_tom log file")
    parser.add_argument("logfile", help="Path to input .txt log file")
    parser.add_argument("-o", "--out", default=None, help="Output CSV (default: <logfile>.csv)")
    args = parser.parse_args()

    out_csv = args.out or (os.path.splitext(args.logfile)[0] + ".csv")
    process_log(args.logfile, out_csv)
