from app import start_tryon
from PIL import Image

def demo_virtual_tryon():
    # Example human and garment images
    example_human_image_path = 'example/human2/00034_00.jpg'  # Update with actual path
    example_garment_image_path = 'example/cloth/04469_00.jpg'  # Update with actual path

    # Load example images
    example_human_image = Image.open(example_human_image_path)
    example_garment_image = Image.open(example_garment_image_path)

    # Set up example dict for human background and optional layers
    example_dict = {
        'background': example_human_image,
        'layers': None  # Optional, add layer images as needed
    }

    # Example inputs for the try-on process
    example_garment_description = "dress"
    example_is_checked = True  # Use auto-generated mask
    example_is_checked_crop = False  # No auto-crop & resizing
    example_denoise_steps = 30
    example_seed = 42

    # Call the start_tryon function with example inputs
    result_image, result_mask = start_tryon(
        dict=example_dict,
        garm_img=example_garment_image,
        garment_des=example_garment_description,
        is_checked=example_is_checked,
        is_checked_crop=example_is_checked_crop,
        denoise_steps=example_denoise_steps,
        seed=example_seed
    )

    # Display the results
    result_image.show()
    result_mask.show()

    # Optionally return the results if needed
    return result_image, result_mask

# To run the demonstration function:
demo_virtual_tryon()