from PIL import Image, ImageDraw, ImageFont
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# --- Convert int to bits ---
def int_to_bits(value, num_bits):
    return [int(bit) for bit in bin(value)[2:].zfill(num_bits)]

# --- Quantum pixel negation ---
def negate_pixel(value, bits, qc, qr, cr, offset=0):
    binary = int_to_bits(value, bits)
    for j in range(bits):
        if binary[bits - 1 - j] == 1:
            qc.x(qr[offset + j])
    qc.barrier()
    for j in range(bits):
        qc.x(qr[offset + j])
    qc.barrier()
    qc.measure(qr[offset:offset + bits], cr[offset:offset + bits])

# --- Save images side by side ---
def save_side_by_side_images(original_img, negated_img, output_path):
    width, height = original_img.size
    combined = Image.new("RGB", (width * 2 + 40, height + 40), (255, 255, 255))

    original_rgb = original_img.convert("RGB")
    negated_rgb = negated_img.convert("RGB")
    combined.paste(original_rgb, (20, 30))
    combined.paste(negated_rgb, (width + 30, 30))

    draw = ImageDraw.Draw(combined)
    font = ImageFont.load_default()
    draw.text((20, 10), "Original", fill=(0, 0, 0), font=font)
    draw.text((width + 30, 10), "Negated", fill=(0, 0, 0), font=font)

    combined.save(output_path)
    print(f"[✔] Side-by-side image saved: {output_path}")

# --- Quantum grayscale image negation ---
def negate_grayscale_image_quantum(image_path):
    img = Image.open(image_path).convert("L")
    width, height = img.size
    matrix = [[img.getpixel((c, r)) for c in range(width)] for r in range(height)]
    intensity_bits = 8
    negated_img = Image.new("L", (width, height))

    first_pixel_circuit = None

    for r in range(height):
        for c in range(width):
            value = matrix[r][c]
            qr = QuantumRegister(intensity_bits, "q")
            cr = ClassicalRegister(intensity_bits, "c")
            qc = QuantumCircuit(qr, cr)
            negate_pixel(value, intensity_bits, qc, qr, cr)

            backend = AerSimulator()
            job = backend.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bitstring = list(counts.keys())[0]

            val = int(bitstring, 2)
            negated_img.putpixel((c, r), val)

            if first_pixel_circuit is None:
                first_pixel_circuit = qc

    return img, negated_img, first_pixel_circuit

# --- Execution ---
if __name__ == "__main__":
    image_path = "lena_grey.jpeg"  # Path to your grayscale image

    print("\nProcessing Grayscale Image...")
    original_img, negated_img, circuit = negate_grayscale_image_quantum(image_path)

    # Save result side-by-side
    save_path = "grayscale_quantum_negated_side_by_side.png"
    save_side_by_side_images(original_img, negated_img, save_path)

    # Print quantum circuit of the first pixel
    print("\nQuantum Circuit for First Pixel:")
    print(circuit.draw(output="text"))
