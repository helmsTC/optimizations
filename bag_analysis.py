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
    
    # Define vibrant colors for better visibility
    colors = [
        '#FF1744',  # Red
        '#00E676',  # Green  
        '#2979FF',  # Blue
        '#FF9100',  # Orange
        '#D500F9',  # Purple
        '#00BFA5',  # Teal
        '#FFD600',  # Yellow
        '#F50057',  # Pink
        '#651FFF',  # Deep Purple
        '#00B8D4',  # Cyan
        '#76FF03',  # Light Green
        '#FF6D00',  # Deep Orange
    ]
    
    # Create figure with two subplots
    print("\n📊 Generating plots...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Calculate overall time range for ideal line
    all_msg_times = []
    all_bag_times = []
    
    for pairs in drift_data.values():
        if pairs:
            msg_times, bag_times = zip(*pairs)
            all_msg_times.extend(msg_times)
            all_bag_times.extend(bag_times)
    
    if all_msg_times:
        time_min = min(min(all_msg_times), min(all_bag_times))
        time_max = max(max(all_msg_times), max(all_bag_times))
    else:
        time_min, time_max = 0, 1
    
    # First plot: Message Time vs Bag Time
    ax1.set_title("Message Timestamp vs Bag Recording Time", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Message Timestamp [s]", fontsize=11)
    ax1.set_ylabel("Bag Recording Time [s]", fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot ideal line (no drift)
    ax1.plot([time_min, time_max], [time_min, time_max], 
             'k--', linewidth=2, alpha=0.5, label='Ideal (No Drift)')
    
    # Second plot: Message Time vs Drift
    ax2.set_title("Timestamp Drift Over Time", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Message Timestamp [s]", fontsize=11)
    ax2.set_ylabel("Drift (Bag Time - Message Time) [s]", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Plot data for each topic
    for idx, (topic, pairs) in enumerate(sorted(drift_data.items())):
        if not pairs:
            continue
        
        msg_times, bag_times = zip(*pairs)
        color = colors[idx % len(colors)]
        
        # Calculate drift statistics
        drifts = [bag - msg for msg, bag in pairs]
        avg_drift = sum(drifts) / len(drifts)
        max_drift = max(abs(d) for d in drifts)
        
        # Simplify topic name for legend
        topic_short = topic.split('/')[-1] if '/' in topic else topic
        label = f"{topic_short} (avg: {avg_drift:.3f}s, max: {max_drift:.3f}s)"
        
        # Plot on first subplot (Message vs Bag time)
        ax1.scatter(msg_times, bag_times, s=2, alpha=0.7, color=color, label=label)
        
        # Plot on second subplot (Message vs Drift)
        ax2.scatter(msg_times, drifts, s=2, alpha=0.7, color=color, label=label)
    
    # Add legends
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
    ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\n📊 Drift Statistics:")
    print("   " + "="*80)
    print(f"   {'Topic':<40} {'Avg Drift':>12} {'Max Drift':>12} {'Status':>10}")
    print("   " + "-"*80)
    
    for topic, pairs in sorted(drift_data.items()):
        if not pairs:
            continue
        
        drifts = [bag - msg for msg, bag in pairs]
        avg_drift = sum(drifts) / len(drifts)
        max_drift = max(abs(d) for d in drifts)
        
        # Determine status
        if abs(avg_drift) < 0.01:
            status = "✅ Excellent"
        elif abs(avg_drift) < 0.1:
            status = "✅ Good"
        elif abs(avg_drift) < 1.0:
            status = "⚠️  Warning"
        else:
            status = "❌ Bad"
        
        print(f"   {topic:<40} {avg_drift:>11.4f}s {max_drift:>11.4f}s {status:>10}")
    
    print("   " + "="*80)
    
    # Print summary
    all_drifts = []
    for pairs in drift_data.values():
        if pairs:
            drifts = [bag - msg for msg, bag in pairs]
            all_drifts.extend(drifts)
    
    if all_drifts:
        overall_avg = sum(all_drifts) / len(all_drifts)
        overall_max = max(abs(d) for d in all_drifts)
        print(f"\n📊 Overall: Average drift = {overall_avg:.4f}s, Max drift = {overall_max:.4f}s")


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
