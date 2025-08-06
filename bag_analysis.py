#!/usr/bin/env python3
"""
debug_bag_messages.py - Debug tool to understand ROS 2 bag message structure
----------------------------------------------------------------------------
This will help us figure out where the actual timestamps are.
"""

import argparse
import os
import sqlite3
import struct
from datetime import datetime


def analyze_bytes_as_timestamp(data: bytes, offset: int, name: str = ""):
    """Try to interpret 8 bytes as a ROS timestamp and show if it makes sense."""
    if len(data) < offset + 8:
        return None
    
    # Try little-endian
    sec_le, nsec_le = struct.unpack_from('<II', data, offset)
    stamp_le = sec_le + nsec_le * 1e-9
    
    # Try big-endian  
    sec_be, nsec_be = struct.unpack_from('>II', data, offset)
    stamp_be = sec_be + nsec_be * 1e-9
    
    # Check if these could be valid timestamps
    results = []
    
    # Check little-endian
    if nsec_le < 1_000_000_000:  # Valid nanoseconds
        if 0 <= stamp_le <= 3600*24*365*10:  # Less than 10 years in seconds (simulation time)
            results.append(f"  LE@{offset:3}: {stamp_le:12.6f}s (sim time: {sec_le}s + {nsec_le}ns)")
        elif 1000000000 <= stamp_le <= 2000000000:  # Unix epoch between 2001 and 2033
            dt = datetime.fromtimestamp(stamp_le)
            results.append(f"  LE@{offset:3}: {stamp_le:12.6f}s (wall: {dt.strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Check big-endian
    if nsec_be < 1_000_000_000:  # Valid nanoseconds
        if 0 <= stamp_be <= 3600*24*365*10:  # Less than 10 years in seconds
            results.append(f"  BE@{offset:3}: {stamp_be:12.6f}s (sim time: {sec_be}s + {nsec_be}ns)")
        elif 1000000000 <= stamp_be <= 2000000000:  # Unix epoch
            dt = datetime.fromtimestamp(stamp_be)
            results.append(f"  BE@{offset:3}: {stamp_be:12.6f}s (wall: {dt.strftime('%Y-%m-%d %H:%M:%S')})")
    
    return results


def debug_message_structure(topic_name: str, msg_bytes: bytes, sample_num: int = 1):
    """Analyze a message to find potential timestamp locations."""
    print(f"\n{'='*70}")
    print(f"Topic: {topic_name} (Sample #{sample_num})")
    print(f"Message size: {len(msg_bytes)} bytes")
    print(f"{'='*70}")
    
    # Show first 64 bytes in hex
    print("\nFirst 64 bytes (hex):")
    for i in range(0, min(64, len(msg_bytes)), 16):
        hex_str = ' '.join(f'{b:02x}' for b in msg_bytes[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in msg_bytes[i:i+16])
        print(f"  {i:04x}: {hex_str:<48} |{ascii_str}|")
    
    # Check for CDR header
    if len(msg_bytes) >= 4:
        cdr_header = struct.unpack_from('<HH', msg_bytes, 0)
        print(f"\nCDR Header: 0x{cdr_header[0]:04x} 0x{cdr_header[1]:04x}")
        if cdr_header[0] & 0xFF == 0x01:
            print("  -> Little-endian CDR")
        elif cdr_header[0] & 0xFF == 0x00:
            print("  -> Big-endian CDR")
    
    # Look for potential timestamps at various offsets
    print("\nSearching for valid timestamps (trying various offsets):")
    found_any = False
    
    # Common offsets to try
    offsets_to_try = [
        (0, "Start of message"),
        (4, "After CDR header"),
        (8, "After CDR + alignment"),
        (12, "After CDR + array length"),
        (16, "After CDR + array length + align"),
    ]
    
    # Also try every 4-byte boundary up to 64 bytes
    for i in range(0, min(64, len(msg_bytes)-7), 4):
        offsets_to_try.append((i, f"Offset {i}"))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_offsets = []
    for offset, desc in offsets_to_try:
        if offset not in seen:
            seen.add(offset)
            unique_offsets.append((offset, desc))
    
    for offset, desc in unique_offsets:
        results = analyze_bytes_as_timestamp(msg_bytes, offset, desc)
        if results:
            if not found_any:
                print(f"\n✅ Potential timestamps found:")
                found_any = True
            print(f"  {desc}:")
            for r in results:
                print(r)
    
    if not found_any:
        print("  ❌ No valid timestamp patterns found in first 64 bytes")
    
    # Special handling for /tf messages
    if '/tf' in topic_name and len(msg_bytes) >= 8:
        print("\n🔍 Special /tf message analysis:")
        # Try to read array length
        array_len_le = struct.unpack_from('<I', msg_bytes, 4)[0]
        array_len_be = struct.unpack_from('>I', msg_bytes, 4)[0]
        print(f"  Array length at offset 4: LE={array_len_le}, BE={array_len_be}")
        
        if 0 < array_len_le < 100:  # Reasonable array size
            print(f"  Likely {array_len_le} transforms in message (LE)")
        elif 0 < array_len_be < 100:
            print(f"  Likely {array_len_be} transforms in message (BE)")


def debug_bag(bag_path: str, max_samples: int = 3):
    """Debug ROS 2 bag to understand message structure."""
    
    # Find database file
    db3 = os.path.join(bag_path, "data_0.db3")
    if not os.path.isfile(db3):
        db3 = os.path.join(bag_path, "data.db3")
        if not os.path.isfile(db3):
            raise FileNotFoundError(f"No .db3 file found in {bag_path}")
    
    print(f"📂 Opening: {db3}")
    
    conn = sqlite3.connect(db3)
    cur = conn.cursor()
    
    # Get topics
    cur.execute("SELECT id, name, type FROM topics")
    topics = cur.fetchall()
    
    print(f"\n📊 Topics in bag:")
    for tid, name, msg_type in topics:
        print(f"  [{tid:2}] {name:40} ({msg_type})")
    
    # Sample a few messages from each topic
    print(f"\n🔍 Analyzing message structure (up to {max_samples} samples per topic)...")
    
    topics_to_analyze = [
        '/clock',      # Should be simple Time message
        '/tf',         # Array of transforms
        '/gt_pose',    # Likely PoseStamped
        '/imu',        # IMU data with header
        '/vision/color'  # Image with header
    ]
    
    for tid, name, msg_type in topics:
        if name not in topics_to_analyze:
            continue
            
        cur.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? LIMIT ?",
            (tid, max_samples)
        )
        messages = cur.fetchall()
        
        for i, (bag_stamp_ns, raw) in enumerate(messages, 1):
            bag_stamp_sec = bag_stamp_ns * 1e-9
            print(f"\n📦 Bag timestamp: {bag_stamp_sec:.6f}s")
            
            # If it's a reasonable bag timestamp, show wall time
            if 1000000000 <= bag_stamp_sec <= 2000000000:
                dt = datetime.fromtimestamp(bag_stamp_sec)
                print(f"   (Wall time: {dt.strftime('%Y-%m-%d %H:%M:%S')})")
            
            debug_message_structure(name, raw, i)
            
            if i >= max_samples:
                break
    
    conn.close()
    print("\n" + "="*70)
    print("Debugging complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug ROS 2 bag message structure to find timestamps"
    )
    parser.add_argument("bag", help="Path to bag directory")
    parser.add_argument("-n", "--samples", type=int, default=2,
                       help="Number of samples per topic to analyze")
    
    args = parser.parse_args()
    
    try:
        debug_bag(args.bag, args.samples)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
