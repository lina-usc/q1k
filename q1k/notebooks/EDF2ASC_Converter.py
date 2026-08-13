#!/usr/bin/env python3
"""
EDF to ASC Converter for EyeLink Files
=======================================
Converts EyeLink EDF files to ASC format for PLR/GO/VEP analysis.

Usage:
    python edf_to_asc.py <path_to_edf_file> [output_directory]

Examples:
    python edf_to_asc.py 1147P_PLR.edf
    python edf_to_asc.py /full/path/to/1147P_GO.edf
    python edf_to_asc.py 1147P_PLR.edf ./my_analysis_folder
    python edf_to_asc.py 1147P_PLR.edf --same  # Place ASC same as EDF location
"""

import sys
import os
from pathlib import Path
import argparse
from datetime import datetime

def convert_edf_to_asc(edf_path, output_dir=None, same_location=False):
    """
    Convert EyeLink EDF file to ASC format
    
    Parameters:
    -----------
    edf_path : str or Path
        Path to the input .edf file
    output_dir : str or Path, optional
        Directory where to save the .asc file
    same_location : bool, optional
        If True, save ASC in the same directory as the EDF file
    """
    
    # Check if eyelinkio is available
    try:
        import eyelinkio
    except ImportError:
        print("❌ eyelinkio not installed. Installing...")
        os.system("pip install eyelinkio")
        import eyelinkio
    
    # Convert to Path object
    edf_path = Path(edf_path)
    
    # Check if EDF file exists
    if not edf_path.exists():
        print(f"❌ Error: EDF file not found: {edf_path}")
        return None
    
    # Determine output path
    if same_location:
        output_path = edf_path.with_suffix('.asc')
    elif output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{edf_path.stem}.asc"
    else:
        # Default: same directory as EDF
        output_path = edf_path.with_suffix('.asc')
    
    # Check if ASC already exists
    if output_path.exists():
        print(f"⚠️  Warning: ASC file already exists: {output_path}")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Skipping conversion.")
            return output_path
    
    print(f"\n{'='*60}")
    print(f"EDF to ASC Converter")
    print(f"{'='*60}")
    print(f"Input EDF:  {edf_path}")
    print(f"Output ASC: {output_path}")
    print(f"File size:  {edf_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"{'='*60}\n")
    
    try:
        # Read EDF file
        print("📖 Reading EDF file...")
        edf_data = eyelinkio.read_edf(str(edf_path))
        
        # Convert to ASC
        print("🔄 Converting to ASC format...")
        edf_data.to_asc(str(output_path))
        
        # Verify conversion
        if output_path.exists():
            asc_size = output_path.stat().st_size / (1024*1024)
            print(f"\n✅ Conversion successful!")
            print(f"   ASC file size: {asc_size:.2f} MB")
            print(f"   Location: {output_path}")
            return output_path
        else:
            print(f"\n❌ Conversion failed: Output file not created")
            return None
            
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        return None

def batch_convert_edf_to_asc(directory, pattern="*.edf", output_dir=None):
    """
    Convert multiple EDF files in a directory
    
    Parameters:
    -----------
    directory : str or Path
        Directory containing EDF files
    pattern : str
        Pattern to match EDF files (default: "*.edf")
    output_dir : str or Path, optional
        Directory for output ASC files (default: same as input)
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return []
    
    edf_files = list(directory.glob(pattern))
    
    if not edf_files:
        print(f"❌ No EDF files found matching '{pattern}' in {directory}")
        return []
    
    print(f"\nFound {len(edf_files)} EDF file(s):")
    for f in edf_files:
        print(f"  - {f.name}")
    
    print(f"\n{'='*60}")
    response = input(f"Convert all {len(edf_files)} files? (y/N): ")
    
    if response.lower() != 'y':
        print("❌ Cancelled.")
        return []
    
    converted = []
    for edf_file in edf_files:
        print(f"\n--- Processing: {edf_file.name} ---")
        result = convert_edf_to_asc(edf_file, output_dir)
        if result:
            converted.append(result)
    
    print(f"\n{'='*60}")
    print(f"✅ Conversion complete! Converted {len(converted)} of {len(edf_files)} files")
    print(f"{'='*60}")
    
    return converted

def main():
    parser = argparse.ArgumentParser(
        description="Convert EyeLink EDF files to ASC format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single file (ASC saved next to EDF)
  python edf_to_asc.py 1147P_PLR.edf
  
  # Convert with specific output directory
  python edf_to_asc.py 1147P_PLR.edf -o ./analysis/PLR/
  
  # Convert and save in same location as EDF
  python edf_to_asc.py 1147P_PLR.edf --same
  
  # Batch convert all EDFs in a directory
  python edf_to_asc.py /path/to/edf/folder --batch --pattern "*PLR*.edf"
  
  # Full path input
  python edf_to_asc.py /full/path/to/1147P_GO.edf -o ./output/
        """
    )
    
    parser.add_argument(
        'input',
        help='Path to EDF file or directory (if --batch is used)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output directory for ASC files (default: same as input)'
    )
    
    parser.add_argument(
        '--same',
        action='store_true',
        help='Save ASC file in the same directory as the EDF file'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Batch mode: input is a directory, convert all EDF files'
    )
    
    parser.add_argument(
        '--pattern',
        default="*.edf",
        help='Pattern for batch conversion (default: "*.edf")'
    )
    
    args = parser.parse_args()
    
    # Batch conversion
    if args.batch:
        batch_convert_edf_to_asc(args.input, args.pattern, args.output)
    # Single file conversion
    else:
        convert_edf_to_asc(args.input, args.output, args.same)

if __name__ == "__main__":
    # If no command-line arguments, run interactive mode
    if len(sys.argv) == 1:
        print("\n" + "="*60)
        print("EDF to ASC Converter - Interactive Mode")
        print("="*60)
        
        # Interactive mode
        edf_path = input("\n📁 Enter path to EDF file: ").strip()
        
        if not edf_path:
            print("❌ No path provided. Exiting.")
            sys.exit(1)
        
        print("\nOutput options:")
        print("  1. Same directory as EDF")
        print("  2. Specific output directory")
        print("  3. Custom location")
        
        choice = input("\nChoose (1/2/3) [default: 1]: ").strip()
        
        if choice == '2':
            output_dir = input("Enter output directory path: ").strip()
            convert_edf_to_asc(edf_path, output_dir)
        elif choice == '3':
            custom_path = input("Enter full output ASC path: ").strip()
            # Create parent directory if needed
            Path(custom_path).parent.mkdir(parents=True, exist_ok=True)
            # Direct conversion with custom path
            import eyelinkio
            edf_data = eyelinkio.read_edf(edf_path)
            edf_data.to_asc(custom_path)
            print(f"✅ Saved to: {custom_path}")
        else:
            convert_edf_to_asc(edf_path, same_location=True)
    else:
        main()