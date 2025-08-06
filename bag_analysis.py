import os
import sqlite3
import argparse
import struct
from collections import defaultdict

import matplotlib.pyplot as plt

def extract_stamp_from_tf_binary(data: bytes):
    """Extract timestamp from serialized TFMessage (assumes 1st transform)."""
    # In TF messages, the first 8 bytes after the array length usually contain the header timestamp.
    # This is highly ROS 2 version dependent — this works *if* only one transform per TFMessage.
    try:
        # Skip initial bytes to get to header.stamp (sec, nanosec)
        offset = 4  # skip array length
        sec, nanosec = struct.unpack_from('<II', data, offset)
        return sec + nanosec * 1e-9
    except struct.error:
        return None

def analyze_bag_tf_only(bag_path):
    db3_file = os.path.join(bag_path, "data_0.db3")
    if not os.path.exists(db3_file):
        raise FileNotFoundError(f"No DB3 file found in: {bag_path}")

    conn = sqlite3.connect(db3_file)
    cursor = conn.cursor()

    # Get topics
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {tid: {"name": name, "type": type_} for tid, name, type_ in cursor.fetchall()}

    time_data = defaultdict(list)

    for topic_id, topic in topics.items():
        name = topic["name"]
        type_ = topic["type"]

        cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id}")
        for bag_time_ns, data in cursor.fetchall():
            msg_time = None

            if "tf2_msgs/msg/TFMessage" in type_ or "/tf" in name:
                msg_time = extract_stamp_from_tf_binary(data)

            if msg_time is None:
                continue  # unsupported or failed to extract

            bag_time = bag_time_ns * 1e-9
            time_data[name].append((msg_time, bag_time))

    # Plot
    plt.figure(figsize=(10, 6))
    for topic, times in time_data.items():
        if not times:
            continue
        x, y = zip(*times)
        plt.plot(x, y, ".", label=topic)

    if any(time_data.values()):
        max_range = max(max(max(x for x, _ in times), max(y for _, y in times)) for times in time_data.values())
        plt.plot([0, max_range], [0, max_range], "k--", label="IDEAL")

    plt.xlabel("Message Timestamp [s]")
    plt.ylabel("Bag Timestamp [s]")
    plt.title("TF Time Comparison (No Deserialization)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Report
    for topic, times in time_data.items():
        diffs = [abs(bag - msg) for msg, bag in times]
        if not diffs:
            continue
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        print(f"Topic: {topic}")
        print(f"  Avg Drift: {avg_diff:.3f}s | Max Drift: {max_diff:.3f}s")
        if avg_diff > 0.5:
            print("  ⚠️ Possible sim time or publishing issue.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to ROS 2 bag directory")
    args = parser.parse_args()

    analyze_bag_tf_only(args.bag_path)
