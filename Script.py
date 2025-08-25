import os
import cv2
import argparse
from inference import get_model
import supervision as sv
from dotenv import load_dotenv  

load_dotenv()


def run_inference_on_image(model, image_path, output_path):
    """Run inference on one image and save annotated result."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_path} (could not load image).")
        return

    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    cv2.imwrite(output_path, annotated_image)
    print(f" Annotated image saved as {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with Roboflow model on images or folders.")
    parser.add_argument("image_path", type=str, help="Path to the input image or folder")
    parser.add_argument("--output", type=str, default="results", help="Folder to save annotated images")
    args = parser.parse_args()

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if api_key is None:
        raise ValueError(" Please set ROBOFLOW_API_KEY in your .env file.")

    model = get_model(model_id="wall-floor-2zskh/1", api_key=api_key)

    if os.path.isdir(args.image_path):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.image_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                input_file = os.path.join(args.image_path, filename)
                output_file = os.path.join(args.output, filename)
                run_inference_on_image(model, input_file, output_file)
    else:
        # Single image
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        run_inference_on_image(model, args.image_path, args.output)


if __name__ == "__main__":
    main()
