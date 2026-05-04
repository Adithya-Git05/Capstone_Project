#!/usr/bin/env python3
"""
Split large zip file into smaller chunks for GitHub upload.
Automatically creates a reassemble script for teammates.
"""

import os
import sys

def split_file(filename: str, chunk_size_mb: int = 90):
    """Split file into chunks."""
    
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    
    file_size = os.path.getsize(filename)
    file_size_mb = file_size / (1024 * 1024)
    chunk_size = chunk_size_mb * 1024 * 1024
    
    print(f"\n{'='*70}")
    print("SPLITTING ZIP FILE")
    print(f"{'='*70}")
    print(f"File: {filename}")
    print(f"Size: {file_size_mb:.2f} MB")
    print(f"Chunk size: {chunk_size_mb} MB")
    
    num_chunks = (file_size + chunk_size - 1) // chunk_size
    print(f"Number of chunks: {num_chunks}\n")
    
    with open(filename, 'rb') as f:
        for i in range(num_chunks):
            chunk_filename = f"{filename}.part{i+1:02d}"
            
            chunk_data = f.read(int(chunk_size))
            chunk_mb = len(chunk_data) / (1024 * 1024)
            
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            print(f"  ✓ Created: {chunk_filename:40s} ({chunk_mb:6.2f} MB)")
    
    print(f"\n✅ Split complete! {num_chunks} chunks created.")
    
    # Create reassemble script
    create_reassemble_script(filename, num_chunks)
    
    print(f"\n{'='*70}")
    print("NEXT STEPS FOR TEAMMATES:")
    print(f"{'='*70}")
    print(f"1. Download all {num_chunks} .part files from GitHub")
    print(f"2. Run the reassemble script:")
    print(f"   python reassemble_zip.py")
    print(f"3. Use the reassembled {filename}")
    print(f"\n{'='*70}")


def create_reassemble_script(original_filename: str, num_chunks: int):
    """Create a reassemble script for teammates."""
    
    script_name = "reassemble_zip.py"
    
    script_content = f'''#!/usr/bin/env python3
"""
Reassemble split zip file chunks back into original file.

Usage:
    python reassemble_zip.py
"""

import os
import sys

def reassemble():
    """Combine all .part files into original zip."""
    
    filename = "{original_filename}"
    num_chunks = {num_chunks}
    
    print("\\n" + "="*70)
    print("REASSEMBLING ZIP FILE")
    print("="*70)
    print(f"Output file: {{filename}}")
    print(f"Number of chunks: {{num_chunks}}\\n")
    
    # Check all chunks exist
    for i in range(num_chunks):
        chunk_file = f"{{filename}}.part{{i+1:02d}}"
        if not os.path.exists(chunk_file):
            print(f"ERROR: Missing chunk: {{chunk_file}}")
            sys.exit(1)
    
    # Combine chunks
    with open(filename, 'wb') as outfile:
        for i in range(num_chunks):
            chunk_file = f"{{filename}}.part{{i+1:02d}}"
            chunk_size_mb = os.path.getsize(chunk_file) / (1024*1024)
            
            with open(chunk_file, 'rb') as infile:
                outfile.write(infile.read())
            
            print(f"  OK Combined: {{chunk_file:40s}} ({{chunk_size_mb:6.2f}} MB)")
    
    output_size_mb = os.path.getsize(filename) / (1024*1024)
    
    print("\\nOK Reassembly complete!")
    print(f"   Output: {{filename}} ({{output_size_mb:.2f}} MB)")
    print(f"\\nYou can now use {{filename}} with dataset.py")
    print("="*70)

if __name__ == "__main__":
    reassemble()
'''
    
    with open(script_name, 'w') as f:
        f.write(script_content)
    
    print(f"\nOK Created reassemble script: {script_name}")


if __name__ == "__main__":
    split_file("working_data.zip", chunk_size_mb=90)
