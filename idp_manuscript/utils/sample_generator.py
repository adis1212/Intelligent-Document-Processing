"""
Sample Image Generator
Creates synthetic manuscript-like test images for the IDP pipeline demo.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random


def generate_sample_manuscripts(output_dir, count=6):
    """
    Generate sample manuscript-style images for testing.
    Creates images with text-like patterns, noise, and varying quality.
    """
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []
    manuscript_texts = [
        "In the ancient scriptures of the Eastern lands,\nwhere wisdom flows like rivers through the sands,\nthe scholars wrote with careful, steady hands.",
        "The Heritage of Knowledge Preserved\nThrough ages dark and times of light,\nthese manuscripts endure the night.",
        "Chapter III: The Art of Calligraphy\nEach stroke a bridge between the past\nand future generations vast.",
        "Ancient records tell of kingdoms grand,\nwhere libraries stretched across the land,\npreserving knowledge, hand to hand.",
        "The Manuscript Archive: Volume VII\nRestored and digitized with care,\na treasure beyond all compare.",
        "Sacred texts from centuries past,\nin fading ink on pages vast,\ntheir wisdom meant forever last.",
    ]

    for i in range(count):
        width, height = 800, 1000
        # Parchment-like background
        bg_color = (
            random.randint(220, 245),
            random.randint(210, 235),
            random.randint(180, 210)
        )
        img = Image.new("RGB", (width, height), bg_color)
        draw = ImageDraw.Draw(img)

        # Add texture/noise
        pixels = np.array(img)
        noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
        pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img)

        # Draw border
        border_color = (random.randint(100, 150), random.randint(80, 120), random.randint(50, 80))
        draw.rectangle([20, 20, width - 20, height - 20], outline=border_color, width=3)
        draw.rectangle([35, 35, width - 35, height - 35], outline=border_color, width=1)

        # Add decorative header line
        draw.line([(40, 80), (width - 40, 80)], fill=border_color, width=2)

        # Add text
        text = manuscript_texts[i % len(manuscript_texts)]
        text_color = (
            random.randint(20, 60),
            random.randint(15, 50),
            random.randint(10, 40)
        )

        try:
            font = ImageFont.truetype("arial.ttf", 22)
            small_font = ImageFont.truetype("arial.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
            small_font = font

        # Title area
        draw.text((50, 50), f"Page {i + 1}", fill=text_color, font=font)

        # Main text body
        y_pos = 100
        for line in text.split("\n"):
            draw.text((50, y_pos), line, fill=text_color, font=font)
            y_pos += 35

        # Add some "aging" effects
        for _ in range(random.randint(3, 8)):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            spot_size = random.randint(5, 20)
            spot_color = (
                bg_color[0] - random.randint(10, 30),
                bg_color[1] - random.randint(10, 30),
                bg_color[2] - random.randint(10, 30),
            )
            draw.ellipse(
                [x1, y1, x1 + spot_size, y1 + spot_size],
                fill=spot_color
            )

        # Page number at bottom
        draw.text(
            (width // 2 - 20, height - 50),
            f"— {i + 1} —",
            fill=text_color,
            font=small_font
        )

        # Introduce quality variations for demo
        if i == 2:
            # Make one image blurry
            img = img.filter(ImageFilter.GaussianBlur(radius=4))
        elif i == 4:
            # Make one image slightly rotated (skewed)
            img = img.rotate(7, fillcolor=bg_color, expand=False)

        filename = f"manuscript_page_{i + 1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, "PNG")
        generated_files.append(filepath)

    return generated_files


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_images")
    files = generate_sample_manuscripts(sample_dir)
    print(f"Generated {len(files)} sample images in {sample_dir}")
    for f in files:
        print(f"  → {os.path.basename(f)}")
