import os
import subprocess
from pathlib import Path

def process_image(input_file, output_dir, method, exe_name):
    # Get the base name of the input file
    base_name = os.path.basename(input_file)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Get absolute paths
    current_dir = os.getcwd()
    exe_dir = os.path.join(current_dir, "exe")
    
    try:
        # Change to exe directory first
        os.chdir(exe_dir)
        
        # Construct the command
        cmd = [
            f"./{exe_name}",
            f"../input/{base_name}",
            f"../output/Result{name_without_ext}.bmp",
            f"{method}"
        ]
        
        print(f"\nProcessing {base_name} using {method} method...")
        print(f"Running command from {os.getcwd()}: {' '.join(cmd)}")
        
        # Run the command
        result = subprocess.run(cmd, check=True)
        print(f"Successfully processed {base_name} with {method}")
        
        # Change back to original directory
        os.chdir(current_dir)
        
        # Add a small delay between processes
        import time
        time.sleep(1)
        
    except Exception as e:
        print(f"Error processing {base_name} with {method}: {e}")
        # Make sure we return to original directory even if there's an error
        os.chdir(current_dir)
    
    print("-" * 50)

def main():
    # Define directories using absolute paths
    current_dir = os.getcwd()
    input_dir = os.path.join(current_dir, "input")
    output_dir = os.path.join(current_dir, "output")
    exe_dir = os.path.join(current_dir, "exe")
    
    # Print directories for debugging
    print(f"Current directory: {current_dir}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Exe directory: {exe_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary mapping methods to their executable names
    method_exes = {
        "sequential": "sequential.exe",
        "omp": "omp_version.exe",
        "mpi": "mpi_version.exe",
        "cuda": "cuda_version.exe"  # Added CUDA version
    }
    
    # Verify all executables exist
    missing_exes = []
    print("\nChecking executables:")
    for method, exe_name in method_exes.items():
        exe_path = os.path.join(exe_dir, exe_name)
        if os.path.exists(exe_path):
            print(f"Found: {exe_name}")
        else:
            print(f"Missing: {exe_name}")
            missing_exes.append(method)
    
    if missing_exes:
        print(f"\nWarning: The following methods are missing executables: {', '.join(missing_exes)}")
        print("Will only process with available methods.")
        # Remove missing methods
        for method in missing_exes:
            del method_exes[method]
    
    # Get all BMP files from input directory
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.bmp')]
    
    if not input_files:
        print("No BMP files found in input directory!")
        return
    
    print(f"\nFound {len(input_files)} BMP files to process")
    print(f"Will process with these methods: {', '.join(method_exes.keys())}")
    print("=" * 50)
    
    # Process each input file with each available method
    for input_file in input_files:
        for method, exe_name in method_exes.items():
            process_image(input_file, output_dir, method, exe_name)

if __name__ == "__main__":
    main()