#!/usr/bin/env python3
"""
analyze_bag_simple.py  –  Drop-in timestamp-drift checker for ROS 2 bags
-----------------------------------------------------------------------

• No ROS imports – works even in a bare Python venv.
• Handles any topic whose message *starts* with std_msgs/Header.
• Handles /tf and /tf_static (first transform's header.stamp).
• Produces a drift plot + console stats.

Usage:
    python analyze_bag_simple.py /path/to/my_bag
"""

import argparse
import os
import sqlite3
import struct
from collections import defaultdict

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------#
#  Low-level helpers
# ---------------------------------------------------------------------#

def _try_extract_stamp(data: bytes, offset: int) -> float | None:
    """Return sec+nanosec from (sec:uint32, nanosec:uint32) at offset, else None."""
    if len(data) < offset + 8:
        return None
    sec, nsec = struct.unpack_from("<II", data, offset)
    # basic sanity check: nsec must be < 1_000_000_000
    if nsec >= 1_000_000_000:
        return None
    # accept either real epoch (>2000) or sim-time small numbers
    return sec + nsec * 1e-9


def extract_message_stamp(topic_name: str, msg_bytes: bytes) -> float | None:
    """
    Heuristic stamp extractor:
      • /tf or /tf_static  -> read at offset 4
      • everything else    -> read at offset 0
    """
    if "/tf" in topic_name:
        return _try_extract_stamp(msg_bytes, 4)  # skip array-length
    return _try_extract_stamp(msg_bytes, 0)


# ---------------------------------------------------------------------#
#  Main analysis routine
# ---------------------------------------------------------------------#

def analyze_bag(bag_path: str) -> None:
    db3 = os.path.join(bag_path, "data_0.db3")
    if not os.path.isfile(db3):
        raise FileNotFoundError(f"Could not find {db3}")

    conn = sqlite3.connect(db3)
    cur = conn.cursor()

    cur.execute("SELECT id, name FROM topics")
    topics = dict(cur.fetchall())               # {topic_id: name}

    # Gather (msg_stamp, bag_stamp) pairs per topic
    drift_data: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)

    for tid, tname in topics.items():
        cur.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (tid,))
        for bag_stamp_ns, raw in cur.fetchall():
            msg_stamp = extract_message_stamp(tname, raw)
            if msg_stamp is None:
                continue
            drift_data[tname].append((msg_stamp, bag_stamp_ns * 1e-9))

    if not drift_data:
        print("❌  No stamped messages could be parsed. "
              "Bag may use custom types whose first field isn't a std_msgs/Header.")
        return

    # ------------------------------------------------------------------#
    #  Plot
    # ------------------------------------------------------------------#
    plt.figure(figsize=(10, 6))

    overall_max = 0.0
    for topic, pairs in drift_data.items():
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        overall_max = max(overall_max, max(xs + ys))
        plt.plot(xs, ys, ".", markersize=3, label=topic)

    plt.plot([0, overall_max], [0, overall_max], "k--", label="IDEAL")
    plt.xlabel("Message timestamp [s]")
    plt.ylabel("Bag recording time [s]")
    plt.title("ROS 2 Bag – Timestamp vs Bag Time")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------#
    #  Console drift stats
    # ------------------------------------------------------------------#
    for topic, pairs in drift_data.items():
        diffs = [abs(bag - msg) for msg, bag in pairs]
        if not diffs:
            continue
        avg_d, max_d = sum(diffs) / len(diffs), max(diffs)
        print(f"{topic:40}  avg drift {avg_d:8.3f}s   max drift {max_d:8.3f}s"
              + ("   ⚠️" if avg_d > 0.5 else ""))


# ---------------------------------------------------------------------#
#  Entrypoint
# ---------------------------------------------------------------------#

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ROS 2 bag timestamp diagnostic (no ROS deps)")
    ap.add_argument("bag", help="Path to the bag directory (containing data_0.db3)")
    analyze_bag(ap.parse_args().bag)
