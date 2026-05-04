#!/usr/bin/env python3
"""
Reassemble split zip file chunks back into original file.

Usage:
    python reassemble_zip.py
"""

import os
import sys

def reassemble():
    """Combine all .part files into original zip."""
    
    filename = "working_data.zip"
    num_chunks = 2
    
    print("\n" + "="*70)
    print("REASSEMBLING ZIP FILE")
    print("="*70)
    print(f"Output file: {filename}")
    print(f"Number of chunks: {num_chunks}\n")
    
    # Check all chunks exist
    for i in range(num_chunks):
        chunk_file = f"{filename}.part{i+1:02d}"
        if not os.path.exists(chunk_file):
            print(f"ERROR: Missing chunk: {chunk_file}")
            sys.exit(1)
    
    # Combine chunks
    with open(filename, 'wb') as outfile:
        for i in range(num_chunks):
            chunk_file = f"{filename}.part{i+1:02d}"
            chunk_size_mb = os.path.getsize(chunk_file) / (1024*1024)
            
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
            
            print(f"  OK Combined: {chunk_file:40s} ({chunk_size_mb:6.2f} MB)")
    
    output_size_mb = os.path.getsize(filename) / (1024*1024)
    
    print("\nOK Reassembly complete!")
    print(f"   Output: {filename} ({output_size_mb:.2f} MB)")
    print(f"\nYou can now use {filename} with dataset.py")
    print("="*70)

if __name__ == "__main__":
    reassemble()
