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
    print(f"[\u2714] Side-by-side image saved: {output_path}")

# --- Quantum image negation ---
def negate_image_quantum(image_path, image_type):
    img = Image.open(image_path)
    width, height = img.size

    if image_type == "binary":
        img = img.convert("L")
        matrix = [[1 if img.getpixel((c, r)) >= 128 else 0 for c in range(width)] for r in range(height)]
        intensity_bits = 1
        negated_img = Image.new("L", (width, height))
    elif image_type == "grayscale":
        img = img.convert("L")
        matrix = [[img.getpixel((c, r)) for c in range(width)] for r in range(height)]
        intensity_bits = 8
        negated_img = Image.new("L", (width, height))
    else:  # colorful
        img = img.convert("RGB")
        matrix = [[img.getpixel((c, r)) for c in range(width)] for r in range(height)]
        intensity_bits = 8
        negated_img = Image.new("RGB", (width, height))

    for r in range(height):
        for c in range(width):
            if image_type == "colorful":
                r_val, g_val, b_val = matrix[r][c]
                total_bits = 24
                qr = QuantumRegister(total_bits, "q")
                cr = ClassicalRegister(total_bits, "c")
                qc = QuantumCircuit(qr, cr)
                negate_pixel(r_val, 8, qc, qr, cr, 0)
                negate_pixel(g_val, 8, qc, qr, cr, 8)
                negate_pixel(b_val, 8, qc, qr, cr, 16)
            else:
                value = matrix[r][c]
                total_bits = intensity_bits
                qr = QuantumRegister(total_bits, "q")
                cr = ClassicalRegister(total_bits, "c")
                qc = QuantumCircuit(qr, cr)
                negate_pixel(value, total_bits, qc, qr, cr)

            # Print circuit only for the first pixel
            if r == 0 and c == 0:
                print(f"Quantum Circuit for {image_type} image first pixel ({r}, {c}):")
                print(qc)

            backend = AerSimulator()
            job = backend.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bitstring = list(counts.keys())[0]

            if image_type == "colorful":
                r_neg = int(bitstring[:8], 2)
                g_neg = int(bitstring[8:16], 2)
                b_neg = int(bitstring[16:], 2)
                negated_img.putpixel((c, r), (r_neg, g_neg, b_neg))
            else:
                val = int(bitstring, 2) * 255 if intensity_bits == 1 else int(bitstring, 2)
                negated_img.putpixel((c, r), val)

    return img, negated_img

# --- Execution ---
if __name__ == "__main__":
    image_paths = {
        "binary": "binaryimage.png",
        "grayscale": "grey.jpeg",
        "colorful": "lena.jpeg"
    }

    for img_type in ["binary", "grayscale", "colorful"]:
        print(f"\nProcessing {img_type.capitalize()} Image...")
        orig_img, neg_img = negate_image_quantum(image_paths[img_type], img_type)
        save_path = f"{img_type}_quantum_negated_side_by_side.png"
        save_side_by_side_images(orig_img, neg_img, save_path)

