import os
import zipfile
import tarfile
from pathlib import Path

def compress_folder_zip(folder_path, output_name="results_compressed.zip"):
    """Compress folder using ZIP with maximum compression."""
    print(f"Compressing {folder_path} to {output_name}...")
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arc_name)
                print(f"Added: {arc_name}")
    
    original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                       for dirpath, dirnames, filenames in os.walk(folder_path)
                       for filename in filenames) / (1024**3)
    
    compressed_size = os.path.getsize(output_name) / (1024**3)
    ratio = (1 - compressed_size/original_size) * 100
    
    print(f"Original: {original_size:.2f} GB")
    print(f"Compressed: {compressed_size:.2f} GB")
    print(f"Compression ratio: {ratio:.1f}%")

def compress_folder_tar_xz(folder_path, output_name="results_compressed.tar.xz"):
    """Compress folder using TAR.XZ (best compression, slower)."""
    print(f"Compressing {folder_path} to {output_name} (this may take a while)...")
    
    with tarfile.open(output_name, 'w:xz', preset=9) as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    
    original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                       for dirpath, dirnames, filenames in os.walk(folder_path)
                       for filename in filenames) / (1024**3)
    
    compressed_size = os.path.getsize(output_name) / (1024**3)
    ratio = (1 - compressed_size/original_size) * 100
    
    print(f"Original: {original_size:.2f} GB")
    print(f"Compressed: {compressed_size:.2f} GB") 
    print(f"Compression ratio: {ratio:.1f}%")

# Usage - choose one method:

# Method 1: ZIP (fast, good compression, widely compatible)
# compress_folder_zip("./results")

# Method 2: TAR.XZ (slower but maximum compression)
compress_folder_tar_xz("./results")

# To decompress later:
# ZIP: zipfile.ZipFile('results_compressed.zip').extractall()
# TAR.XZ: tarfile.open('results_compressed.tar.xz').extractall()