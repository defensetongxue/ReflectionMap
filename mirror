from PIL import Image, ImageFilter
import numpy as np
import random

def mirror(image_path, save_path):
    # Load and resize the image
    img = Image.open(image_path)
    img = img.resize((224, 224))

    # Determine the starting point of the crop
    mean_begin = 65
    std_begin = 15
    begin = max(30, min(int(np.random.normal(mean_begin, std_begin)), 90))

    # Determine the length of the crop
    length = random.randint(113, 133)

    # Ensure the crop does not exceed image bounds
    end = min(begin + length, 224)

    # Crop the middle section of the image
    middle_part = img.crop((0, begin, 224, end))

    # Copy and flip the cropped part
    flipped_part = middle_part.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Apply Gaussian blur to the flipped part
    guass_norm=random.randint(0, 2)
    blurred_part = flipped_part.filter(ImageFilter.GaussianBlur(radius=guass_norm))

    # Concatenate the original cropped part, the flipped blurred part, and the remaining part
    final_img = Image.new('RGB', (224, 224))
    final_img.paste(middle_part, (0, 0))
    final_img.paste(blurred_part, (0, middle_part.height))

    # Calculate remaining space and fill it
    remaining_space = 224 - (middle_part.height + blurred_part.height)
    if remaining_space > 0:
        additional_part = middle_part.crop((0, 0, 224, remaining_space))
        final_img.paste(additional_part, (0, middle_part.height + blurred_part.height))

    # Save the final image
    final_img.save(save_path)
