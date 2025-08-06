#!/usr/bin/env python3
"""
analyze_bag_easy.py - Simple ROS 2 bag timestamp analyzer using rosbags library
------------------------------------------------------------------------------
No ROS installation required! Just: pip install rosbags matplotlib
"""

import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


def get_timestamp_from_message(msg) -> float | None:
    """Extract timestamp from a message if it has a header."""
    try:
        # Check for header field (most sensor messages)
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Check for transforms (tf messages)
        if hasattr(msg, 'transforms') and len(msg.transforms) > 0:
            transform = msg.transforms[0]
            if hasattr(transform, 'header') and hasattr(transform.header, 'stamp'):
                return transform.header.stamp.sec + transform.header.stamp.nanosec * 1e-9
        
        # Check for clock messages
        if hasattr(msg, 'clock'):
            return msg.clock.sec + msg.clock.nanosec * 1e-9
            
        # Direct timestamp (for Time messages)
        if hasattr(msg, 'sec') and hasattr(msg, 'nanosec'):
            return msg.sec + msg.nanosec * 1e-9
            
    except Exception:
        pass
    
    return None


def analyze_bag(bag_path: str):
    """Analyze ROS 2 bag for timestamp drift using rosbags library."""
    
    print(f"📂 Opening bag: {bag_path}")
    
    # Store drift data
    drift_data = defaultdict(list)
    message_counts = defaultdict(int)
    extracted_counts = defaultdict(int)
    
    # Read the bag
    with Reader(bag_path) as reader:
        print(f"📊 Duration: {reader.duration * 1e-9:.2f} seconds")
        print(f"📊 Messages: {reader.message_count}")
        
        # Print topics
        print("\n📋 Topics:")
        for connection in reader.connections:
            topic = connection.topic
            msgtype = connection.msgtype
            count = connection.msgcount
            print(f"   • {topic:40} [{msgtype:30}] ({count} messages)")
        
        print("\n⏳ Processing messages...")
        
        # Process messages
        for connection, timestamp, rawdata in reader.messages():
            topic = connection.topic
            message_counts[topic] += 1
            
            # Deserialize the message
            try:
                msg = deserialize_cdr(rawdata, connection.msgtype)
                msg_stamp = get_timestamp_from_message(msg)
                
                if msg_stamp is not None:
                    bag_stamp = timestamp * 1e-9  # Convert from nanoseconds
                    drift_data[topic].append((msg_stamp, bag_stamp))
                    extracted_counts[topic] += 1
                    
            except Exception as e:
                # Skip messages that can't be deserialized
                continue
    
    # Print extraction statistics
    print("\n📈 Extraction Statistics:")
    for topic in sorted(message_counts.keys()):
        total = message_counts[topic]
        extracted = extracted_counts[topic]
        if total > 0:
            success_rate = (extracted / total) * 100
            status = "✅" if success_rate > 50 else "⚠️" if success_rate > 0 else "❌"
            print(f"   {status} {topic:40} {extracted:5}/{total:5} messages "
                  f"({success_rate:.1f}% success)")
    
    if not drift_data:
        print("\n❌ No timestamped messages found.")
        return
    
    # Create plot
    print("\n📊 Generating plot...")
    plt.figure(figsize=(12, 7))
    
    for topic, pairs in drift_data.items():
        if not pairs:
            continue
        
        msg_times, bag_times = zip(*pairs)
        
        # Calculate drift
        drifts = [bag - msg for msg, bag in pairs]
        avg_drift = sum(drifts) / len(drifts)
        max_drift = max(abs(d) for d in drifts)
        
        # Plot
        plt.scatter(msg_times, drifts, s=1, alpha=0.5, 
                   label=f"{topic} (avg: {avg_drift:.3f}s, max: {max_drift:.3f}s)")
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='No drift')
    plt.xlabel("Message Timestamp [s]")
    plt.ylabel("Drift (Bag Time - Message Time) [s]")
    plt.title("ROS 2 Bag - Timestamp Drift Analysis")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n📊 Drift Statistics:")
    print("   " + "="*70)
    print(f"   {'Topic':<40} {'Avg Drift':>12} {'Max Drift':>12}")
    print("   " + "-"*70)
    
    for topic, pairs in sorted(drift_data.items()):
        if not pairs:
            continue
        
        drifts = [bag - msg for msg, bag in pairs]
        avg_drift = sum(drifts) / len(drifts)
        max_drift = max(abs(d) for d in drifts)
        
        status = "✅" if abs(avg_drift) < 0.1 else "⚠️" if abs(avg_drift) < 1.0 else "❌"
        print(f"   {topic:<40} {avg_drift:>11.4f}s {max_drift:>11.4f}s {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Easy ROS 2 bag timestamp analyzer using rosbags library"
    )
    parser.add_argument("bag", help="Path to bag directory")
    args = parser.parse_args()
    
    try:
        analyze_bag(args.bag)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
