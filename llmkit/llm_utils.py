import base64
from pathlib import Path

from loguru import logger


def image_path_to_b64_string(
    image_path: str, encoding: str = "ascii", image_format: str = "png"
) -> str:
    """
    Encodes an image file into a base64-encoded data URL string.

    Steps:
      1. Verify the image file exists.
      2. Read file bytes and encode in base64.
      3. Construct and return the data URL.

    Args:
        image_path (str): Path to the image file.
        encoding (str): String encoding to use (default is 'ascii').
        image_format (str): Format of the image, used in the data URI (default is 'png').

    Returns:
        str: Base64-encoded image as a data URL.
    """
    logger.debug(f" Encoding image to base64, path={image_path}, format={image_format}")
    path = Path(image_path)
    if not path.is_file():
        logger.error(f" Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bytes: bytes = path.read_bytes()
    encoded_bytes: str = base64.b64encode(image_bytes).decode(encoding)
    logger.debug(f" Encoded image size: {len(encoded_bytes)} characters")
    return f"data:image/{image_format};base64,{encoded_bytes}"
