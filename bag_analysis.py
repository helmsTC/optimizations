import os
import sqlite3
import argparse
import struct
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_tf_transforms(data: bytes):
    """Extract list of (frame_id, child_frame_id, timestamp) from TFMessage binary."""
    transforms = []

    try:
        offset = 0
        (array_len,) = struct.unpack_from('<I', data, offset)
        offset += 4

        for _ in range(array_len):
            # Get timestamp
            sec, nanosec = struct.unpack_from('<II', data, offset)
            msg_time = sec + nanosec * 1e-9
            offset += 8

            # Extract frame_id string
            (frame_id_len,) = struct.unpack_from('<I', data, offset)
            offset += 4
            frame_id = data[offset:offset+frame_id_len].decode('utf-8')
            offset += frame_id_len

            # Extract child_frame_id string (after pose)
            offset += 56  # skip transform (translation + rotation)
            (child_id_len,) = struct.unpack_from('<I', data, offset)
            offset += 4
            child_frame_id = data[offset:offset+child_id_len].decode('utf-8')
            offset += child_frame_id_len

            label = f"{frame_id} TO {child_frame_id}"
            transforms.append((label, msg_time))
    except Exception:
        pass

    return transforms

def extract_header_stamp(data: bytes):
    """Extract header.stamp.sec + .nanosec from messages like PoseStamped (first 8 bytes after header)."""
    try:
        offset = 0
        sec, nanosec = struct.unpack_from('<II', data, offset)
        return sec + nanosec * 1e-9
    except Exception:
        return None

def analyze_bag(bag_path):
    db3_file = os.path.join(bag_path, "data_0.db3")
    if not os.path.exists(db3_file):
        raise FileNotFoundError(f"No DB3 file found in: {bag_path}")

    conn = sqlite3.connect(db3_file)
    cursor = conn.cursor()

    # Get topics
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {tid: {"name": name, "type": type_} for tid, name, type_ in cursor.fetchall()}

    # Store series: {label: [(msg_time, bag_time)]}
    time_data = defaultdict(list)

    for topic_id, topic in topics.items():
        name = topic["name"]
        type_ = topic["type"]

        cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id}")
        for bag_time_ns, data in cursor.fetchall():
            bag_time = bag_time_ns * 1e-9

            if "tf2_msgs/msg/TFMessage" in type_ or "/tf" in name:
                transforms = extract_tf_transforms(data)
                for label, msg_time in transforms:
                    time_data[f"/tf {label}"].append((msg_time, bag_time))

            elif "PoseStamped" in type_ or "geometry_msgs" in type_ or "/lidar" in name:
                msg_time = extract_header_stamp(data)
                if msg_time:
                    time_data[name].append((msg_time, bag_time))

    if not time_data:
        print("❌ No timestamp data found in bag.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    for label, points in time_data.items():
        if not points:
            continue
        x, y = zip(*points)
        plt.plot(x, y, ".", label=label)

    max_t = max(max(max(x for x, _ in pts), max(y for _, y in pts)) for pts in time_data.values())
    plt.plot([0, max_t], [0, max_t], "k--", label="IDEAL")

    plt.xlabel("Message Timestamp [s]")
    plt.ylabel("Bag Timestamp [s]")
    plt.title("Message vs Bag Time (Grouped by Frame or Topic)")
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Report drift
    for label, points in time_data.items():
        diffs = [abs(b - m) for m, b in points]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        print(f"{label}")
        print(f"  Avg Drift: {avg_diff:.3f}s | Max Drift: {max_diff:.3f}s")
        if avg_diff > 0.5:
            print("  ⚠️ High drift — possible time sync or sim_time issue")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to ROS 2 bag directory")
    args = parser.parse_args()

    analyze_bag(args.bag_path)
