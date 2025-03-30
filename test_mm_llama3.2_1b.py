import torch

"""
 - aten.mm                                             31.823T     98.48%
                                                        1.473T      4.56%  ['(1372, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(1, 2048):torch.bfloat16']
                                                        1.473T      4.56%  ['(8192, 1372)(1, 8192):torch.bfloat16', '(1372, 2048)(2048, 1):torch.bfloat16']
                                                        1.473T      4.56%  ['(1372, 8192)(8192, 1):torch.bfloat16', '(8192, 2048)(2048, 1):torch.bfloat16']
                                                        1.246T      3.85%  ['(1160, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(1, 2048):torch.bfloat16']
                                                        1.246T      3.85%  ['(8192, 1160)(1, 8192):torch.bfloat16', '(1160, 2048)(2048, 1):torch.bfloat16']
                                                        1.246T      3.85%  ['(1160, 8192)(8192, 1):torch.bfloat16', '(8192, 2048)(2048, 1):torch.bfloat16']
                                                        0.737T      2.28%  ['(1372, 8192)(8192, 1):torch.bfloat16', '(8192, 2048)(1, 8192):torch.bfloat16']
                                                        0.737T      2.28%  ['(2048, 1372)(1, 2048):torch.bfloat16', '(1372, 8192)(8192, 1):torch.bfloat16']
                                                        0.737T      2.28%  ['(1372, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(8192, 1):torch.bfloat16']
                                                        0.696T      2.15%  ['(648, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(1, 2048):torch.bfloat16']
                                                        0.696T      2.15%  ['(8192, 648)(1, 8192):torch.bfloat16', '(648, 2048)(2048, 1):torch.bfloat16']
                                                        0.696T      2.15%  ['(648, 8192)(8192, 1):torch.bfloat16', '(8192, 2048)(2048, 1):torch.bfloat16']
                                                        0.674T      2.09%  ['(628, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(1, 2048):torch.bfloat16']
                                                        0.674T      2.09%  ['(8192, 628)(1, 8192):torch.bfloat16', '(628, 2048)(2048, 1):torch.bfloat16']
                                                        0.674T      2.09%  ['(628, 8192)(8192, 1):torch.bfloat16', '(8192, 2048)(2048, 1):torch.bfloat16']
                                                        0.633T      1.96%  ['(172, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.633T      1.96%  ['(128256, 172)(1, 128256):torch.bfloat16', '(172, 2048)(2048, 1):torch.bfloat16']
                                                        0.633T      1.96%  ['(172, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.623T      1.93%  ['(2048, 1160)(1, 2048):torch.bfloat16', '(1160, 8192)(8192, 1):torch.bfloat16']
                                                        0.623T      1.93%  ['(1160, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(8192, 1):torch.bfloat16']
                                                        0.544T      1.68%  ['(148, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.544T      1.68%  ['(128256, 148)(1, 128256):torch.bfloat16', '(148, 2048)(2048, 1):torch.bfloat16']
                                                        0.544T      1.68%  ['(148, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.520T      1.61%  ['(8192, 484)(1, 8192):torch.bfloat16', '(484, 2048)(2048, 1):torch.bfloat16']
                                                        0.520T      1.61%  ['(484, 8192)(8192, 1):torch.bfloat16', '(8192, 2048)(2048, 1):torch.bfloat16']
                                                        0.368T      1.14%  ['(2048, 1372)(1, 2048):torch.bfloat16', '(1372, 2048)(2048, 1):torch.bfloat16']
                                                        0.368T      1.14%  ['(1372, 2048)(2048, 1):torch.bfloat16', '(2048, 2048)(2048, 1):torch.bfloat16']
                                                        0.348T      1.08%  ['(2048, 648)(1, 2048):torch.bfloat16', '(648, 8192)(8192, 1):torch.bfloat16']
                                                        0.348T      1.08%  ['(648, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(8192, 1):torch.bfloat16']
                                                        0.337T      1.04%  ['(2048, 628)(1, 2048):torch.bfloat16', '(628, 8192)(8192, 1):torch.bfloat16']
                                                        0.337T      1.04%  ['(628, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(8192, 1):torch.bfloat16']
                                                        0.311T      0.96%  ['(2048, 1160)(1, 2048):torch.bfloat16', '(1160, 2048)(2048, 1):torch.bfloat16']
                                                        0.311T      0.96%  ['(1160, 2048)(2048, 1):torch.bfloat16', '(2048, 2048)(2048, 1):torch.bfloat16']
                                                        0.309T      0.96%  ['(84, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.309T      0.96%  ['(128256, 84)(1, 128256):torch.bfloat16', '(84, 2048)(2048, 1):torch.bfloat16']
                                                        0.309T      0.96%  ['(84, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.294T      0.91%  ['(80, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.294T      0.91%  ['(128256, 80)(1, 128256):torch.bfloat16', '(80, 2048)(2048, 1):torch.bfloat16']
                                                        0.294T      0.91%  ['(80, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.260T      0.80%  ['(2048, 484)(1, 2048):torch.bfloat16', '(484, 8192)(8192, 1):torch.bfloat16']
                                                        0.260T      0.80%  ['(484, 2048)(2048, 1):torch.bfloat16', '(2048, 8192)(8192, 1):torch.bfloat16']
                                                        0.235T      0.73%  ['(64, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.235T      0.73%  ['(128256, 64)(1, 128256):torch.bfloat16', '(64, 2048)(2048, 1):torch.bfloat16']
                                                        0.235T      0.73%  ['(64, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.174T      0.54%  ['(2048, 648)(1, 2048):torch.bfloat16', '(648, 2048)(2048, 1):torch.bfloat16']
                                                        0.174T      0.54%  ['(648, 2048)(2048, 1):torch.bfloat16', '(2048, 2048)(2048, 1):torch.bfloat16']
                                                        0.169T      0.52%  ['(2048, 628)(1, 2048):torch.bfloat16', '(628, 2048)(2048, 1):torch.bfloat16']
                                                        0.169T      0.52%  ['(628, 2048)(2048, 1):torch.bfloat16', '(2048, 2048)(2048, 1):torch.bfloat16']
                                                        0.130T      0.40%  ['(2048, 484)(1, 2048):torch.bfloat16', '(484, 2048)(2048, 1):torch.bfloat16']
                                                        0.130T      0.40%  ['(484, 2048)(2048, 1):torch.bfloat16', '(2048, 2048)(2048, 1):torch.bfloat16']
                                                        0.092T      0.28%  ['(512, 1372)(1, 512):torch.bfloat16', '(1372, 2048)(2048, 1):torch.bfloat16']
                                                        0.092T      0.28%  ['(1372, 512)(512, 1):torch.bfloat16', '(512, 2048)(2048, 1):torch.bfloat16']
                                                        0.088T      0.27%  ['(168, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.088T      0.27%  ['(128256, 168)(1, 128256):torch.bfloat16', '(168, 2048)(2048, 1):torch.bfloat16']
                                                        0.088T      0.27%  ['(168, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.078T      0.24%  ['(512, 1160)(1, 512):torch.bfloat16', '(1160, 2048)(2048, 1):torch.bfloat16']
                                                        0.078T      0.24%  ['(1160, 512)(512, 1):torch.bfloat16', '(512, 2048)(2048, 1):torch.bfloat16']
                                                        0.065T      0.20%  ['(124, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.065T      0.20%  ['(128256, 124)(1, 128256):torch.bfloat16', '(124, 2048)(2048, 1):torch.bfloat16']
                                                        0.065T      0.20%  ['(124, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.043T      0.13%  ['(512, 648)(1, 512):torch.bfloat16', '(648, 2048)(2048, 1):torch.bfloat16']
                                                        0.043T      0.13%  ['(648, 512)(512, 1):torch.bfloat16', '(512, 2048)(2048, 1):torch.bfloat16']
                                                        0.042T      0.13%  ['(512, 628)(1, 512):torch.bfloat16', '(628, 2048)(2048, 1):torch.bfloat16']
                                                        0.042T      0.13%  ['(628, 512)(512, 1):torch.bfloat16', '(512, 2048)(2048, 1):torch.bfloat16']
                                                        0.036T      0.11%  ['(68, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.036T      0.11%  ['(128256, 68)(1, 128256):torch.bfloat16', '(68, 2048)(2048, 1):torch.bfloat16']
                                                        0.036T      0.11%  ['(68, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.032T      0.10%  ['(512, 484)(1, 512):torch.bfloat16', '(484, 2048)(2048, 1):torch.bfloat16']
                                                        0.032T      0.10%  ['(484, 512)(512, 1):torch.bfloat16', '(512, 2048)(2048, 1):torch.bfloat16']
                                                        0.032T      0.10%  ['(60, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.032T      0.10%  ['(128256, 60)(1, 128256):torch.bfloat16', '(60, 2048)(2048, 1):torch.bfloat16']
                                                        0.032T      0.10%  ['(60, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
                                                        0.019T      0.06%  ['(36, 2048)(2048, 1):torch.bfloat16', '(2048, 128256)(1, 2048):torch.bfloat16']
                                                        0.019T      0.06%  ['(128256, 36)(1, 128256):torch.bfloat16', '(36, 2048)(2048, 1):torch.bfloat16']
                                                        0.019T      0.06%  ['(36, 128256)(128256, 1):torch.bfloat16', '(128256, 2048)(2048, 1):torch.bfloat16']
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
        (((1372, 2048), (2048, 1), 'torch.bfloat16', (2048, 8192), (1, 2048), 'torch.bfloat16'), 4.56),
        (((8192, 1372), (1, 8192), 'torch.bfloat16', (1372, 2048), (2048, 1), 'torch.bfloat16'), 4.56),
        (((1372, 8192), (8192, 1), 'torch.bfloat16', (8192, 2048), (2048, 1), 'torch.bfloat16'), 4.56),
        (((1160, 2048), (2048, 1), 'torch.bfloat16', (2048, 8192), (1, 2048), 'torch.bfloat16'), 3.85),
        (((8192, 1160), (1, 8192), 'torch.bfloat16', (1160, 2048), (2048, 1), 'torch.bfloat16'), 3.85),
        (((1160, 8192), (8192, 1), 'torch.bfloat16', (8192, 2048), (2048, 1), 'torch.bfloat16'), 3.85),
        (((1372, 8192), (8192, 1), 'torch.bfloat16', (8192, 2048), (1, 8192), 'torch.bfloat16'), 2.28),
        (((2048, 1372), (1, 2048), 'torch.bfloat16', (1372, 8192), (8192, 1), 'torch.bfloat16'), 2.28),
        (((1372, 2048), (2048, 1), 'torch.bfloat16', (2048, 8192), (8192, 1), 'torch.bfloat16'), 2.28),
        (((648, 2048), (2048, 1), 'torch.bfloat16', (2048, 8192), (1, 2048), 'torch.bfloat16'), 2.15),
        (((8192, 648), (1, 8192), 'torch.bfloat16', (648, 2048), (2048, 1), 'torch.bfloat16'), 2.15),
        (((648, 8192), (8192, 1), 'torch.bfloat16', (8192, 2048), (2048, 1), 'torch.bfloat16'), 2.15),
        (((628, 2048), (2048, 1), 'torch.bfloat16', (2048, 8192), (1, 2048), 'torch.bfloat16'), 2.09),
        (((8192, 628), (1, 8192), 'torch.bfloat16', (628, 2048), (2048, 1), 'torch.bfloat16'), 2.09),
        (((628, 8192), (8192, 1), 'torch.bfloat16', (8192, 2048), (2048, 1), 'torch.bfloat16'), 2.09),
        (((172, 2048), (2048, 1), 'torch.bfloat16', (2048, 128256), (1, 2048), 'torch.bfloat16'), 1.96),
        (((128256, 172), (1, 128256), 'torch.bfloat16', (172, 2048), (2048, 1), 'torch.bfloat16'), 1.96),
        (((172, 128256), (128256, 1), 'torch.bfloat16', (128256, 2048), (2048, 1), 'torch.bfloat16'), 1.96),
        (((2048, 1160), (1, 2048), 'torch.bfloat16', (1160, 8192), (8192, 1), 'torch.bfloat16'), 1.93),
        (((1160, 2048), (2048, 1), 'torch.bfloat16', (2048, 8192), (8192, 1), 'torch.bfloat16'), 1.93),
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
