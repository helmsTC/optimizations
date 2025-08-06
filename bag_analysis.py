import os
import sqlite3
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

def analyze_bag(bag_path):
    db_file = os.path.join(bag_path, "metadata.yaml")
    db3_file = os.path.join(bag_path, "data_0.db3")
    if not os.path.exists(db3_file):
        raise FileNotFoundError(f"No DB3 file found in: {bag_path}")

    conn = sqlite3.connect(db3_file)
    cursor = conn.cursor()

    # Get topic info
    cursor.execute("SELECT id, name, type FROM topics")
    topics = {tid: {"name": name, "type": type_} for tid, name, type_ in cursor.fetchall()}

    # Store (message_timestamp, bag_timestamp) pairs
    time_data = defaultdict(list)

    for topic_id in topics:
        cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id}")
        messages = cursor.fetchall()
        for bag_time_ns, data in messages:
            # Assume timestamp is in the header
            # Extract seconds and nanoseconds from raw serialized msg
            # For simplicity, we approximate:
            # - bag_time in seconds
            # - msg_time in seconds from embedded data (dummy parse)
            try:
                # In production, use `rosbag2_py` and deserialize properly
                from rclpy.serialization import deserialize_message
                from std_msgs.msg import Header
                import rosidl_runtime_py.utilities as rosidl_utils

                msg_type = rosidl_utils.get_message_class(topics[topic_id]["type"])
                msg = deserialize_message(data, msg_type)
                msg_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            except Exception:
                # Can't parse the message properly
                continue

            bag_time = bag_time_ns * 1e-9
            time_data[topics[topic_id]["name"]].append((msg_time, bag_time))

    # Plot
    plt.figure(figsize=(10, 6))
    for topic, times in time_data.items():
        x, y = zip(*times)
        plt.plot(x, y, ".", label=topic)

    plt.plot([0, max(max(x) for x, y in time_data.values())], 
             [0, max(max(y) for x, y in time_data.values())], 
             "k--", label="IDEAL")

    plt.xlabel("Message Timestamp [s]")
    plt.ylabel("Bag Timestamp [s]")
    plt.title("ROS 2 Bag Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Flag time drifts
    for topic, times in time_data.items():
        diffs = [abs(bag - msg) for msg, bag in times]
        avg_diff = sum(diffs) / len(diffs) if diffs else 0
        max_diff = max(diffs) if diffs else 0
        print(f"Topic: {topic}")
        print(f"  Average Timestamp Drift: {avg_diff:.3f}s")
        print(f"  Max Timestamp Drift:     {max_diff:.3f}s")
        if avg_diff > 0.5:
            print("  ⚠️ WARNING: High average drift. Check sim time usage.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to the ROS 2 bag directory")
    args = parser.parse_args()

    analyze_bag(args.bag_path)
