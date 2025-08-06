import os
import sqlite3
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt

def analyze_bag(bag_path):
    import rclpy
    from rclpy.serialization import deserialize_message
    import rosidl_runtime_py.utilities as rosidl_utils

    rclpy.init()

    db_file = os.path.join(bag_path, "data_0.db3")
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"No DB3 file found in: {bag_path}")

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Get topic info
    cursor.execute("SELECT id, name, type FROM topics")
    topic_info = {tid: {"name": name, "type": type_} for tid, name, type_ in cursor.fetchall()}

    time_data = defaultdict(list)
    failed_topics = set()

    for topic_id, info in topic_info.items():
        topic_name = info["name"]
        topic_type = info["type"]

        try:
            msg_class = rosidl_utils.get_message_class(topic_type)
        except Exception:
            print(f"⚠️ Skipping {topic_name} (can't load message class for type {topic_type})")
            failed_topics.add(topic_name)
            continue

        cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id}")
        for bag_time_ns, data in cursor.fetchall():
            try:
                msg = deserialize_message(data, msg_class)

                # Try to extract header timestamp
                if hasattr(msg, 'header'):
                    stamp = msg.header.stamp
                    msg_time = stamp.sec + stamp.nanosec * 1e-9
                elif hasattr(msg, 'transforms'):  # TFMessage
                    if len(msg.transforms) == 0:
                        continue
                    stamp = msg.transforms[0].header.stamp
                    msg_time = stamp.sec + stamp.nanosec * 1e-9
                else:
                    continue  # Skip messages without timestamp

                bag_time = bag_time_ns * 1e-9
                time_data[topic_name].append((msg_time, bag_time))

            except Exception as e:
                continue  # Skip corrupted messages

    if not time_data:
        print("❌ No valid time data found. Nothing to analyze.")
        return

    # Plot
    plt.figure(figsize=(10, 6))
    for topic, times in time_data.items():
        if not times:
            continue
        x, y = zip(*times)
        plt.plot(x, y, ".", label=topic)

    try:
        max_x = max(max(x for x, _ in times) for times in time_data.values() if times)
        max_y = max(max(y for _, y in times) for times in time_data.values() if times)
        plt.plot([0, max(max_x, max_y)], [0, max(max_x, max_y)], "k--", label="IDEAL")
    except ValueError:
        print("⚠️ Could not determine max X/Y for IDEAL line.")

    plt.xlabel("Message Timestamp [s]")
    plt.ylabel("Bag Timestamp [s]")
    plt.title("ROS 2 Bag Time Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Report drift
    for topic, times in time_data.items():
        if not times:
            continue
        diffs = [abs(bag - msg) for msg, bag in times]
        avg_diff = sum(diffs) / len(diffs)
        max_diff = max(diffs)
        print(f"Topic: {topic}")
        print(f"  Avg Drift: {avg_diff:.3f}s | Max Drift: {max_diff:.3f}s")
        if avg_diff > 0.5:
            print("  ⚠️ Potential clock or sim_time issue.")

    if failed_topics:
        print("\n⚠️ The following topics were skipped due to missing message types:")
        for topic in failed_topics:
            print(f"  - {topic}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_path", help="Path to the ROS 2 bag directory")
    args = parser.parse_args()

    analyze_bag(args.bag_path)
