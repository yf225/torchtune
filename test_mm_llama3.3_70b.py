"""
How to collect NCU metrics for mm kernels:
1. Run the llama model with command: `wp tune run --nproc_per_node 8 full_finetune_distributed --config llama3_3/70B_full >output.txt 2>&1` (make sure full_finetune_distributed.py is using our custom FlopCounterMode),
2. Extract the mm kernel shapes from the log file with `python ~/extract_kernel_shapes.py <kernel_name> output.txt >kernel_shapes.csv 2>&1` (optionally only extract top-N-flops kernels),
3. Put the mm kernel shapes into this file (run_mm_operations()) and run the script, to get the combined CSV file.
4. Run `python ~/ncu_trace_parser.py <combined_CSV_file>` to get the NCU metrics summary.
"""

import torch
from collections import defaultdict
import csv
from io import StringIO
from datetime import datetime

def create_tensor_with_strides(shape, stride, dtype, device):
    # Create tensor with specific shape and strides
    tensor = torch.randn(*shape, dtype=dtype, device=device)
    tensor.as_strided(shape, stride)
    return tensor

def create_run_script(shape1, stride1, type1, shape2, stride2, type2, script_path):
    """Create a run script for a specific matrix multiplication shape"""
    # Convert dtype strings to actual type names (e.g., 'torch.bfloat16' -> 'bfloat16')
    dtype1_str = type1.split('.')[-1]
    dtype2_str = type2.split('.')[-1]
    
    script_content = f"""
import torch

def run_mm():
    device = torch.device("cuda")
    dtype1 = torch.{dtype1_str}
    dtype2 = torch.{dtype2_str}
    
    # Create tensors with specific shapes and strides
    mat1 = torch.randn({shape1}, dtype=dtype1, device=device)
    mat1 = mat1.as_strided({shape1}, {stride1})
    
    mat2 = torch.randn({shape2}, dtype=dtype2, device=device)
    mat2 = mat2.as_strided({shape2}, {stride2})
    
    # Warmup
    for _ in range(5):
        result = torch.mm(mat1, mat2)
    
    # Actual run
    result = torch.mm(mat1, mat2)
    
if __name__ == "__main__":
    run_mm()
"""
    with open(script_path, 'w') as f:
        f.write(script_content)
    return script_path

def run_ncu_trace(kernel_id, script_path):
    """Run NCU trace for a specific kernel ID and script"""
    import subprocess
    import os
    from datetime import datetime
    import csv
    from io import StringIO
    
    # Create the full command that sources bashrc and runs the command
    cmd = f"""
source ~/.bashrc
export CUDA_INJECTION64_PATH=none
dyno dcgm_profiling --mute=true --duration=100000_s >/dev/null 2>&1

# Define metrics to collect
METRICS="dram__bytes.sum.per_second,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,group:memory__shared_table"

# Generate the report file with specific metrics (suppress output)
$SUDO ${{CUDA_HOME}}/bin/ncu \\
    --metrics "$METRICS" \\
    --import-source yes \\
    -o "{kernel_id}_profile" \\
    -f \\
    python {script_path} >/dev/null 2>&1

# Convert to CSV with the same metrics (suppress output)
${{CUDA_HOME}}/bin/ncu -i "{kernel_id}_profile.ncu-rep" \\
    --csv \\
    --page raw \\
    --metrics "$METRICS" >"{kernel_id}_metrics.csv" 2>/dev/null

# Only show the CSV contents
cat "{kernel_id}_metrics.csv"
rm -rf "{kernel_id}_profile.ncu-rep"
rm -rf "{kernel_id}_metrics.csv"
"""
    
    try:
        # Use bash to execute the commands
        result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Error running NCU trace for kernel {kernel_id}: {e}")
        return None
    finally:
        # Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)

def convert_to_gbytes(value, current_unit):
    """Convert value to GBytes/s from either TBytes/s or GBytes/s"""
    value = float(value)
    if "TByte/s".lower() in current_unit.lower():
        return value * 1024  # Convert TBytes to GBytes
    elif "GByte/s".lower() in current_unit.lower():
        return value
    else:
        raise ValueError(f"Unexpected unit {current_unit}")

def run_mm_operations():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Each entry is now a tuple of ((shape1, stride1, type1, shape2, stride2, type2), flop%)
    shapes_with_percentages = [
        # Rank 1 (Global: 230.899T; aten.mm: 230.284T)
        (((272, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 17.71),
        (((174, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 11.33),
        (((28672, 272), (272, 1), 'torch.bfloat16', (272, 8192), (1, 272), 'torch.bfloat16'), 8.85),
        (((272, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 8.85),
        (((28672, 174), (174, 1), 'torch.bfloat16', (174, 8192), (1, 174), 'torch.bfloat16'), 5.66),

        # Rank 2 (Global: 218.423T; aten.mm: 217.892T)
        (((234, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 16.10),
        (((188, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 12.94),
        (((28672, 234), (234, 1), 'torch.bfloat16', (234, 8192), (1, 234), 'torch.bfloat16'), 8.05),
        (((234, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 8.05),
        (((28672, 188), (188, 1), 'torch.bfloat16', (188, 8192), (1, 188), 'torch.bfloat16'), 6.47),

        # Rank 3 (Global: 341.271T; aten.mm: 339.746T)
        (((474, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 20.88),
        (((28672, 474), (474, 1), 'torch.bfloat16', (474, 8192), (1, 474), 'torch.bfloat16'), 10.44),
        (((474, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 10.44),
        (((184, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 8.10),
        (((474, 8192), (8192, 1), 'torch.bfloat16', (8192, 8192), (1, 8192), 'torch.bfloat16'), 5.97),

        # Rank 4 (Global: 189.375T; aten.mm: 188.977T)
        (((198, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 15.72),
        (((168, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 13.34),
        (((28672, 198), (198, 1), 'torch.bfloat16', (198, 8192), (1, 198), 'torch.bfloat16'), 7.86),
        (((198, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 7.86),
        (((28672, 168), (168, 1), 'torch.bfloat16', (168, 8192), (1, 168), 'torch.bfloat16'), 6.67),

        # Rank 5 (Global: 209.171T; aten.mm: 208.598T)
        (((290, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 20.84),
        (((28672, 290), (290, 1), 'torch.bfloat16', (290, 8192), (1, 290), 'torch.bfloat16'), 10.42),
        (((290, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 10.42),
        (((114, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 8.19),
        (((290, 8192), (8192, 1), 'torch.bfloat16', (8192, 8192), (1, 8192), 'torch.bfloat16'), 5.95),

        # Rank 6 (Global: 164.492T; aten.mm: 164.193T)
        (((164, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 14.99),
        (((154, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 14.07),
        (((28672, 164), (164, 1), 'torch.bfloat16', (164, 8192), (1, 164), 'torch.bfloat16'), 7.49),
        (((164, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 7.49),
        (((28672, 154), (154, 1), 'torch.bfloat16', (154, 8192), (1, 154), 'torch.bfloat16'), 7.04),

        # Rank 7 (Global: 244.417T; aten.mm: 243.708T)
        (((302, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 18.57),
        (((170, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 10.46),
        (((28672, 302), (302, 1), 'torch.bfloat16', (302, 8192), (1, 302), 'torch.bfloat16'), 9.29),
        (((302, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 9.29),
        (((302, 8192), (8192, 1), 'torch.bfloat16', (8192, 8192), (1, 8192), 'torch.bfloat16'), 5.31),

        # Rank 8 (Global: 288.007T; aten.mm: 287.080T)
        (((314, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 16.39),
        (((242, 8192), (8192, 1), 'torch.bfloat16', (8192, 28672), (1, 8192), 'torch.bfloat16'), 12.63),
        (((28672, 314), (314, 1), 'torch.bfloat16', (314, 8192), (1, 314), 'torch.bfloat16'), 8.19),
        (((314, 28672), (28672, 1), 'torch.bfloat16', (28672, 8192), (1, 28672), 'torch.bfloat16'), 8.19),
        (((28672, 242), (242, 1), 'torch.bfloat16', (242, 8192), (1, 242), 'torch.bfloat16'), 6.32),
    ]
    
    results = []
    all_csv_data = []
    header = None
    unit_row = None
    
    for idx, ((shape1, stride1, type1, shape2, stride2, type2), percentage) in enumerate(shapes_with_percentages):
        kernel_id = f"mm_kernel_{idx}"
        script_path = f"run_mm_{idx}.py"
        
        # Create run script for this shape with new parameters
        create_run_script(shape1, stride1, type1, shape2, stride2, type2, script_path)
        
        # Run NCU trace and collect CSV output
        csv_output = run_ncu_trace(kernel_id, script_path)
        if csv_output:
            csv_reader = csv.reader(StringIO(csv_output))
            csv_data = list(csv_reader)
            assert len(csv_data) >= 3
            
            current_header = csv_data[0]
            current_units = csv_data[1]
            
            # Store header from first CSV
            if header is None:
                header = current_header
                unit_row = current_units[:]
                # Force DRAM bandwidth unit to GByte/s
                dram_col_idx = header.index("dram__bytes.sum.per_second")
                unit_row[dram_col_idx] = "GByte/s"
                # Add FLOP% column
                header.append("FLOP%")
                unit_row.append("%")
                all_csv_data.extend([header, unit_row])
            
            # Convert and add data rows
            for row in csv_data[2:]:
                dram_col_idx = header.index("dram__bytes.sum.per_second")
                # Convert the DRAM bandwidth value to GBytes/s
                row[dram_col_idx] = f"{convert_to_gbytes(row[dram_col_idx], current_units[dram_col_idx]):.2f}"
                row.append(f"{percentage:.2f}")  # Add FLOP%
                all_csv_data.append(row)
        
        try:
            # Create tensors with specific strides
            mat1 = create_tensor_with_strides(shape1, stride1, eval(type1), device)
            mat2 = create_tensor_with_strides(shape2, stride2, eval(type2), device)
            result = torch.mm(mat1, mat2)
            results.append(result)
            print(f"Successfully computed mm for shapes {shape1} x {shape2} -> {result.shape} (Combined {percentage:.2f}% of FLOPs)")
        except RuntimeError as e:
            print(f"Failed for shapes {shape1} x {shape2} (Combined {percentage:.2f}% of FLOPs): {e}")

    # Save combined CSV data to file with timestamp
    timestamp = int(datetime.now().timestamp())
    output_file = f"{timestamp}_metrics.csv"
    with open(output_file, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(all_csv_data)

    # Print summary of deduplication
    print("\nShape deduplication summary:")
    print(f"# of shapes: {len(shapes_with_percentages)}")

    print(f"\nSaved combined metrics to: {output_file}")

if __name__ == "__main__":
    run_mm_operations() 
