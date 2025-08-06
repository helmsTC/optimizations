#!/usr/bin/env python3
"""
analyze_bag_improved.py - Improved ROS 2 bag timestamp analyzer with CDR support
--------------------------------------------------------------------------------
Handles CDR encapsulation and alignment properly.
"""

import argparse
import os
import sqlite3
import struct
from collections import defaultdict
from typing import Optional, Tuple

import matplotlib.pyplot as plt


def read_cdr_header(data: bytes, offset: int = 0) -> Tuple[int, int]:
    """
    Read CDR encapsulation header (4 bytes).
    Returns (endianness, next_offset).
    """
    if len(data) < offset + 4:
        return 0, offset
    
    # CDR header: 2 bytes for representation, 2 bytes options
    repr_id, options = struct.unpack_from('<HH', data, offset)
    
    # Check endianness (0x0001 = little endian CDR, 0x0000 = big endian CDR)
    is_little = (repr_id & 0xFF) == 0x01
    
    return 1 if is_little else 0, offset + 4


def align_offset(offset: int, alignment: int) -> int:
    """Align offset to the specified alignment boundary."""
    remainder = offset % alignment
    if remainder == 0:
        return offset
    return offset + (alignment - remainder)


def extract_timestamp_from_cdr(data: bytes, skip_array_length: bool = False) -> Optional[float]:
    """
    Extract timestamp from CDR-encoded message data.
    
    Args:
        data: Raw message bytes
        skip_array_length: True for /tf messages (need to skip 4-byte array length)
    
    Returns:
        Timestamp in seconds or None if extraction failed
    """
    try:
        # Read CDR header
        endian, offset = read_cdr_header(data, 0)
        endian_char = '<' if endian else '>'
        
        # Skip array length for /tf messages
        if skip_array_length:
            if len(data) < offset + 4:
                return None
            # Read array length (typically for transforms[] array)
            array_len = struct.unpack_from(f'{endian_char}I', data, offset)[0]
            offset += 4
            
            # Sanity check array length
            if array_len == 0 or array_len > 1000:  # arbitrary upper limit
                return None
        
        # Align to 8-byte boundary for the timestamp structure
        offset = align_offset(offset, 8)
        
        # Read timestamp (sec: uint32, nanosec: uint32)
        if len(data) < offset + 8:
            return None
            
        sec, nsec = struct.unpack_from(f'{endian_char}II', data, offset)
        
        # Validate nanoseconds
        if nsec >= 1_000_000_000:
            # Try alternate offset in case of alignment issues
            offset_alt = offset + 4
            if len(data) >= offset_alt + 8:
                sec, nsec = struct.unpack_from(f'{endian_char}II', data, offset_alt)
                if nsec >= 1_000_000_000:
                    return None
        
        # Convert to seconds
        return sec + nsec * 1e-9
        
    except Exception as e:
        # Uncomment for debugging
        # print(f"Error extracting timestamp: {e}")
        return None


def extract_message_stamp(topic_name: str, msg_bytes: bytes) -> Optional[float]:
    """
    Extract timestamp from message based on topic type.
    
    Args:
        topic_name: Name of the ROS topic
        msg_bytes: Raw message data
    
    Returns:
        Timestamp in seconds or None
    """
    # Special handling for tf messages
    if topic_name in ['/tf', '/tf_static']:
        return extract_timestamp_from_cdr(msg_bytes, skip_array_length=True)
    
    # Standard header messages
    return extract_timestamp_from_cdr(msg_bytes, skip_array_length=False)


def analyze_bag(bag_path: str, verbose: bool = False) -> None:
    """
    Analyze ROS 2 bag for timestamp drift.
    
    Args:
        bag_path: Path to bag directory
        verbose: Enable verbose output
    """
    # Find the database file
    db3 = os.path.join(bag_path, "data_0.db3")
    if not os.path.isfile(db3):
        # Try alternate naming
        db3 = os.path.join(bag_path, "data.db3")
        if not os.path.isfile(db3):
            raise FileNotFoundError(
                f"Could not find database file in {bag_path}\n"
                f"Expected 'data_0.db3' or 'data.db3'"
            )
    
    print(f"📂 Opening bag: {db3}")
    
    conn = sqlite3.connect(db3)
    cur = conn.cursor()
    
    # Get topic information
    try:
        cur.execute("SELECT id, name FROM topics")
        topics = dict(cur.fetchall())
    except sqlite3.OperationalError as e:
        print(f"❌ Error reading topics table: {e}")
        print("   This may not be a valid ROS 2 bag database.")
        return
    
    print(f"📊 Found {len(topics)} topics")
    if verbose:
        for tid, name in topics.items():
            print(f"   • {name}")
    
    # Gather (msg_stamp, bag_stamp) pairs per topic
    drift_data = defaultdict(list)
    failed_extractions = defaultdict(int)
    total_messages = defaultdict(int)
    
    for tid, tname in topics.items():
        cur.execute("SELECT timestamp, data FROM messages WHERE topic_id = ?", (tid,))
        messages = cur.fetchall()
        
        for bag_stamp_ns, raw in messages:
            total_messages[tname] += 1
            msg_stamp = extract_message_stamp(tname, raw)
            
            if msg_stamp is None:
                failed_extractions[tname] += 1
                continue
                
            # Convert bag timestamp from nanoseconds to seconds
            bag_stamp_sec = bag_stamp_ns * 1e-9
            drift_data[tname].append((msg_stamp, bag_stamp_sec))
    
    # Print extraction statistics
    print("\n📈 Extraction Statistics:")
    for topic in sorted(total_messages.keys()):
        total = total_messages[topic]
        failed = failed_extractions[topic]
        success = total - failed
        if total > 0:
            success_rate = (success / total) * 100
            status = "✅" if success_rate > 50 else "⚠️" if success_rate > 0 else "❌"
            print(f"   {status} {topic:40} {success:5}/{total:5} messages "
                  f"({success_rate:.1f}% success rate)")
    
    if not drift_data:
        print("\n❌ No timestamped messages could be extracted.")
        print("   Possible causes:")
        print("   • Messages don't have std_msgs/Header as first field")
        print("   • Non-standard CDR encoding")
        print("   • Custom message types")
        return
    
    # Plot results
    print("\n📊 Generating drift plot...")
    plt.figure(figsize=(12, 7))
    
    overall_max = 0.0
    colors = plt.cm.tab10(range(10))
    
    for idx, (topic, pairs) in enumerate(drift_data.items()):
        if not pairs:
            continue
            
        msg_times, bag_times = zip(*pairs)
        overall_max = max(overall_max, max(msg_times + bag_times))
        
        # Use different colors for different topics
        color = colors[idx % len(colors)]
        plt.plot(msg_times, bag_times, '.', markersize=3, 
                label=f"{topic} ({len(pairs)} msgs)", color=color, alpha=0.7)
    
    # Add ideal line
    plt.plot([0, overall_max], [0, overall_max], 'k--', 
             linewidth=1, label="Ideal (no drift)", alpha=0.5)
    
    plt.xlabel("Message Timestamp [s]")
    plt.ylabel("Bag Recording Time [s]")
    plt.title("ROS 2 Bag - Timestamp Drift Analysis")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    
    # Add text box with statistics
    stats_text = "Drift Statistics:\n"
    for topic, pairs in sorted(drift_data.items()):
        if not pairs:
            continue
        drifts = [abs(bag - msg) for msg, bag in pairs]
        avg_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        stats_text += f"{topic.split('/')[-1]}: avg={avg_drift:.3f}s, max={max_drift:.3f}s\n"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.show()
    
    # Print drift statistics
    print("\n📊 Timestamp Drift Analysis:")
    print("   " + "="*70)
    print(f"   {'Topic':<40} {'Avg Drift':>12} {'Max Drift':>12} {'Status':>8}")
    print("   " + "-"*70)
    
    for topic, pairs in sorted(drift_data.items()):
        if not pairs:
            continue
            
        drifts = [abs(bag - msg) for msg, bag in pairs]
        avg_drift = sum(drifts) / len(drifts)
        max_drift = max(drifts)
        
        # Determine status based on drift
        if avg_drift < 0.01:
            status = "✅ Good"
        elif avg_drift < 0.1:
            status = "⚠️  OK"
        elif avg_drift < 1.0:
            status = "⚠️  High"
        else:
            status = "❌ Bad"
        
        print(f"   {topic:<40} {avg_drift:>11.4f}s {max_drift:>11.4f}s  {status}")
    
    print("   " + "="*70)
    
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ROS 2 bag timestamp drift analyzer (no ROS dependencies)"
    )
    parser.add_argument("bag", help="Path to the bag directory containing .db3 file")
    parser.add_argument("-v", "--verbose", action="store_true", 
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        analyze_bag(args.bag, args.verbose)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
