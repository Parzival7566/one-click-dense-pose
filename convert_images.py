import cv2
from densepose.utils.helper import GetLogger, Predictor
import os

def process_image(input_path, output_path):
    logger = GetLogger.logger(__name__)
    predictor = Predictor()

    frame = cv2.imread(input_path)  # Read the image

    if frame is None:
        logger.error(f"Failed to read image: {input_path}")
        return None

    out_frame, out_frame_seg = predictor.predict(frame)

    # Save the processed image
    cv2.imwrite(output_path, out_frame_seg)

    logger.info(f"Processing complete for image: {input_path}")  # Report completion
    return out_frame_seg

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", type=str, help="Set the input image file path", required=True
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Set the output image file path",
        required=True,
    )
    args = parser.parse_args()

    process_image(args.input_path, args.output_path)
