import cv2
from utils.helper import GetLogger, Predictor
from argparse import ArgumentParser
import sys
import os

parser = ArgumentParser()
parser.add_argument(
    "--input_dir", type=str, help="Set the input directory of images", required=True
)
parser.add_argument(
    "--out_dir",
    type=str,
    help="Set the output directory for processed images",
    required=True,
)
args = parser.parse_args()

logger = GetLogger.logger(__name__)
predictor = Predictor()

# Input/output directories
input_dir = args.input_dir
output_dir = args.out_dir

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get list of image files in the input directory
image_files = [
    f for f in os.listdir(input_dir) if f.endswith((".jpg", ".jpeg", ".png"))
]
n_frames = len(image_files)
logger.info(f"No of frames {n_frames}")

# Process each image 
done = 0
for image_file in image_files:
    input_path = os.path.join(input_dir, image_file)
    frame = cv2.imread(input_path)  # Read the image

    out_frame, out_frame_seg = predictor.predict(frame)

    # Construct output path
    output_filename = os.path.splitext(image_file)[0] + "_processed.jpg"  # Or choose a different naming scheme
    output_path = os.path.join(output_dir, output_filename)

    # Save the processed image
    cv2.imwrite(output_path, out_frame_seg)

    done += 1
    percent = int((done / n_frames) * 100)
    sys.stdout.write(
        "\rProgress: [{}{}] {}%".format("=" * percent, " " * (100 - percent), percent)
    )
    sys.stdout.flush()

logger.info("Processing complete!")  # Report completion
