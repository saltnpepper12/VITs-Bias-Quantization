import pandas as pd
import os
from PIL import Image
from io import BytesIO
import requests

# Path to your downloaded FairFace validation parquet file
parquet_path = "validation-00000-of-00001-09e3e67bb00ab4ec.parquet"
output_dir = "calibration_images"
os.makedirs(output_dir, exist_ok=True)

# Load the parquet file
df = pd.read_parquet(parquet_path)

print("Columns in the parquet file:", df.columns)

count = 0
for idx, row in df.iterrows():
    # Try common column names for image data
    image_info = None
    for col in ['file', 'image_path', 'image_url']:
        if col in row and pd.notnull(row[col]):
            image_info = row[col]
            break
    
    img = None
    if image_info is not None:
        # If it's a URL, download; if it's a path, open locally
        if str(image_info).startswith('http'):
            try:
                response = requests.get(image_info)
                img = Image.open(BytesIO(response.content))
            except Exception as e:
                print(f"Failed to download image at {image_info}: {e}")
                continue
        else:
            try:
                img = Image.open(image_info)
            except Exception as e:
                print(f"Failed to open image at {image_info}: {e}")
                continue
    elif 'image' in row and row['image'] is not None:
        # If the image is stored as a dict (common in Hugging Face datasets)
        try:
            img_data = row['image']
            if isinstance(img_data, dict):
                # Try common keys
                if 'bytes' in img_data:
                    img = Image.open(BytesIO(img_data['bytes']))
                elif 'array' in img_data:
                    img = Image.open(BytesIO(img_data['array']))
                elif 'path' in img_data:
                    img = Image.open(img_data['path'])
                else:
                    print(f"Unknown image dict keys at row {idx}: {img_data.keys()}")
                    continue
            elif isinstance(img_data, bytes):
                img = Image.open(BytesIO(img_data))
            else:
                print(f"Unknown image data type at row {idx}: {type(img_data)}")
                continue
        except Exception as e:
            print(f"Failed to decode image at row {idx}: {e}")
            continue
    else:
        print(f"No image found at row {idx}")
        continue
    
    # Save the image
    try:
        img.save(os.path.join(output_dir, f"calib_{count:03d}.jpg"))
        count += 1
    except Exception as e:
        print(f"Failed to save image at row {idx}: {e}")
        continue
    if count >= 100:
        break

print(f"Saved {count} images to {output_dir}")