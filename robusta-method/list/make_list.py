input_file = "XD-Violence-i3d-test.list"
output_file = "XD-Violence-i3d-test-processed.list"

# Dictionary to store unique annotations
annotations = {}

with open(input_file, "r") as infile:
    for line in infile:
        parts = line.strip().split(" | ")
        base_annotation = parts[0].rsplit("__", 1)[0]  # Remove __X.npy
        if base_annotation not in annotations:
            annotations[base_annotation] = parts[1] if len(parts) > 1 else ""

with open(output_file, "w") as outfile:
    for annotation, extra in annotations.items():
        if extra:
            outfile.write(f"{annotation}.npy | {extra}\n")
        else:
            outfile.write(f"{annotation}.npy\n")

print(f"Processed list saved to {output_file}")
