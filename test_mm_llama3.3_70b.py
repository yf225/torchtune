import torch
from collections import defaultdict

def create_tensor_with_strides(shape, stride, dtype, device):
    # Create tensor with specific shape and strides
    tensor = torch.randn(*shape, dtype=dtype, device=device)
    tensor.as_strided(shape, stride)
    return tensor

def deduplicate_shapes(shapes_with_percentages):
    # Dictionary to store unique shapes and sum their percentages
    unique_shapes = defaultdict(float)

    # Sum percentages for identical shape pairs
    for (shape1, shape2), percentage in shapes_with_percentages:
        shape_key = (tuple(shape1), tuple(shape2))
        unique_shapes[shape_key] += percentage

    # Convert back to list of tuples with combined percentages
    return [(list(shapes), percentage) for shapes, percentage in unique_shapes.items()]

def run_mm_operations():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Each entry is now a tuple of ((shape1, shape2), percentage)
    shapes_with_percentages = [
        # Batch 1 (>10%)
        (((474, 8192), (8192, 28672)), 20.88),
        (((314, 8192), (8192, 28672)), 16.39),
        (((272, 8192), (8192, 28672)), 17.71),
        (((242, 8192), (8192, 28672)), 12.63),
        (((174, 8192), (8192, 28672)), 11.33),
        (((184, 8192), (8192, 28672)), 8.10),

        # Batch 2 (5-10%)
        (((28672, 474), (474, 8192)), 10.44),
        (((474, 28672), (28672, 8192)), 10.44),
        (((28672, 314), (314, 8192)), 8.19),
        (((314, 28672), (28672, 8192)), 8.19),
        (((28672, 272), (272, 8192)), 8.85),
        (((272, 28672), (28672, 8192)), 8.85),
        (((28672, 242), (242, 8192)), 6.32),
        (((242, 28672), (28672, 8192)), 6.32),
        (((28672, 174), (174, 8192)), 5.66),
        (((174, 28672), (28672, 8192)), 5.66),

        # Batch 3 (3-5%)
        (((474, 8192), (8192, 8192)), 4.68),
        (((314, 8192), (8192, 8192)), 4.68),
        (((272, 8192), (8192, 8192)), 5.06),
        (((242, 8192), (8192, 8192)), 3.61),
        (((184, 8192), (8192, 8192)), 2.32),
        (((174, 8192), (8192, 8192)), 3.24),

        # Batch 4 (2-3%)
        (((474, 28672), (28672, 8192)), 2.34),
        (((8192, 474), (474, 28672)), 2.34),
        (((474, 8192), (8192, 28672)), 2.34),
        (((314, 28672), (28672, 8192)), 4.10),
        (((8192, 314), (314, 28672)), 4.10),
        (((314, 8192), (8192, 28672)), 4.10),
        (((242, 28672), (28672, 8192)), 3.16),
        (((8192, 242), (242, 28672)), 3.16),
        (((242, 8192), (8192, 28672)), 3.16),

        # Batch 5 (1-2%)
        (((8192, 474), (474, 8192)), 2.34),
        (((474, 8192), (8192, 8192)), 2.34),
        (((8192, 314), (314, 8192)), 2.34),
        (((314, 8192), (8192, 8192)), 2.34),
        (((8192, 242), (242, 8192)), 1.80),
        (((242, 8192), (8192, 8192)), 1.80),
        (((8192, 184), (184, 8192)), 1.16),
        (((184, 8192), (8192, 8192)), 1.16),
        (((8192, 174), (174, 8192)), 1.62),
        (((174, 8192), (8192, 8192)), 1.62),

        # Batch 6 (0.5-1%)
        (((474, 8192), (8192, 1024)), 0.75),
        (((314, 8192), (8192, 1024)), 0.66),
        (((272, 8192), (8192, 1024)), 0.63),
        (((242, 8192), (8192, 1024)), 0.45),
        (((184, 8192), (8192, 1024)), 0.29),
        (((174, 8192), (8192, 1024)), 0.37),

        # Batch 7 (0.2-0.5%)
        (((1024, 474), (474, 8192)), 0.29),
        (((474, 1024), (1024, 8192)), 0.29),
        (((1024, 314), (314, 8192)), 0.33),
        (((314, 1024), (1024, 8192)), 0.33),
        (((1024, 242), (242, 8192)), 0.23),
        (((242, 1024), (1024, 8192)), 0.23),

        # Batch 8 (Small operations <0.2%)
        (((60, 8192), (8192, 128256)), 0.20),
        (((128256, 60), (60, 8192)), 0.20),
        (((60, 128256), (128256, 8192)), 0.20),
        (((40, 8192), (8192, 128256)), 0.20),
        (((128256, 40), (40, 8192)), 0.20),
        (((40, 128256), (128256, 8192)), 0.20),
        (((38, 8192), (8192, 128256)), 0.23),
        (((128256, 38), (38, 8192)), 0.23),
        (((38, 128256), (128256, 8192)), 0.23),
        (((34, 8192), (8192, 128256)), 0.02),
        (((128256, 34), (34, 8192)), 0.02),
        (((34, 128256), (128256, 8192)), 0.02),
        (((32, 8192), (8192, 128256)), 0.16),
        (((128256, 32), (32, 8192)), 0.16),
        (((32, 128256), (128256, 8192)), 0.16),
        (((24, 8192), (8192, 128256)), 0.13),
        (((128256, 24), (24, 8192)), 0.13),
        (((24, 128256), (128256, 8192)), 0.13),
        (((22, 8192), (8192, 128256)), 0.13),
        (((128256, 22), (22, 8192)), 0.13),
        (((22, 128256), (128256, 8192)), 0.13),
        (((18, 8192), (8192, 128256)), 0.01),
        (((128256, 18), (18, 8192)), 0.01),
        (((18, 128256), (128256, 8192)), 0.01),
        (((16, 8192), (8192, 128256)), 0.01),
        (((128256, 16), (16, 8192)), 0.01),
        (((16, 128256), (128256, 8192)), 0.01),
        (((14, 8192), (8192, 128256)), 0.02),
        (((128256, 14), (14, 8192)), 0.02),
        (((14, 128256), (128256, 8192)), 0.02),
        (((10, 8192), (8192, 128256)), 0.01),
        (((128256, 10), (10, 8192)), 0.01),
        (((10, 128256), (128256, 8192)), 0.01),
        (((2, 8192), (8192, 128256)), 0.00),
        (((128256, 2), (2, 8192)), 0.00),
        (((2, 128256), (128256, 8192)), 0.00)
    ]

    # Deduplicate shapes and combine their percentages
    unique_shapes = deduplicate_shapes(shapes_with_percentages)

    # Sort by percentage in descending order
    unique_shapes.sort(key=lambda x: x[1], reverse=True)

    results = []
    # Process each unique shape pair
    for (shape1, shape2), percentage in unique_shapes:
        # For matrix multiplication, stride is typically (shape[1], 1) for first matrix
        # and (1, shape[0]) for second matrix to ensure proper memory layout
        stride1 = (shape1[1], 1)
        stride2 = (1, shape2[0])

        try:
            mat1 = create_tensor_with_strides(shape1, stride1, dtype, device)
            mat2 = create_tensor_with_strides(shape2, stride2, dtype, device)
            result = torch.mm(mat1, mat2)
            results.append(result)
            print(f"Successfully computed mm for shapes {shape1} x {shape2} -> {result.shape} (Combined {percentage:.2f}% of FLOPs)")
        except RuntimeError as e:
            print(f"Failed for shapes {shape1} x {shape2} (Combined {percentage:.2f}% of FLOPs): {e}")

    # Print summary of deduplication
    print("\nShape deduplication summary:")
    print(f"Original number of shapes: {len(shapes_with_percentages)}")
    print(f"Number of unique shapes: {len(unique_shapes)}")
    print("\nTop 10 most compute-intensive unique shapes:")
    for (shape1, shape2), percentage in unique_shapes[:10]:
        print(f"{shape1} x {shape2}: {percentage:.2f}% of FLOPs")

    return results

if __name__ == "__main__":
    run_mm_operations()
