import os
import cv2
from PIL import Image
import pandas as pd
from pathlib import Path

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

def is_valid_image_file(file_path):
    """Check if Image can be Opened and is not corrupt."""
    try:
        # check 1 : Check PIL can open the image and verify
        img = Image.open(file_path)
        img.verify()
        
        # check 2 : Check OpenCV can read the image
        img_cv = cv2.imread(str(file_path))
        if img_cv is None:
            return False , "OpenCV Failed to load"
        
        # Minimu size check 
        h,w = img_cv.shape[:2]
        if h < 10 or w < 10:
            return False, f"Suspiciously small image dimensions: {w}x{h}"
        return True, "Valid Image"
    except Exception as e:
            return False, f"Error: {str(e)}"
    

def validate_dataset(root_dir,split="train"):
     """Walk through root_dir/split/<class>/*.jpg and Validate each image and returns a DataFrame report """
     split_path = Path(root_dir) / split
     records = []
     for class_folder in sorted(split_path.iterdir()):
          if not class_folder.is_dir():
               continue
          class_name  = class_folder.name
          for img_file in class_folder.iterdir():
               if img_file.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                    records.append({
                         "file_path": str(img_file),
                         "class":class_name,
                         "valid":False,
                         "reason":f"Invalid File extension: {img_file.suffix}"
                    })
                    continue
               valid , reason = is_valid_image_file(img_file)
               records.append({
                     "file_path": str(img_file),
                     "class":class_name,
                     "valid":valid,
                     "reason":reason
                })
     df = pd.DataFrame(records)
     return df

if __name__ == "__main__":
     ROOT_DIR = "data/raw/Animal"
     for split in ["train","val"]:
          print(f"Validating {split} set...")
          report = validate_dataset(ROOT_DIR,split=split)
          os.makedirs("reports", exist_ok=True)
          report.to_csv(f"reports/{split}_validation_report.csv", index=False)

          total_images = len(report)
          invalid_images = (~report["valid"]).sum()
          print(f"{split}:{total_images} images, {invalid_images} invalid images found")
          if invalid_images > 0:
               print(report[~report["valid"]][["file_path","reason"]])