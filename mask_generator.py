import json
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt


def decode_rle(rle, height, width):

    mask = np.zeros(height * width, dtype=np.uint8)
    rle_pairs = np.array(rle).reshape(-1, 2)
    
    for start, length in rle_pairs:
        start -= 1  
        mask[start:start+length] = 1
        
    return mask.reshape((height, width))


with open('result.json', 'r') as f:
    coco_data = json.load(f)


image_path = 'images/7297f39a-20240911_080133.jpg'
image = Image.open(image_path).convert("RGB")
image = ImageOps.exif_transpose(image)



binary_mask = np.zeros((image.height, image.width), dtype=np.uint8)


image_id = 0
annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]


for ann in annotations:
    segmentation = ann['segmentation']
    
    
    if isinstance(segmentation, list):
        for poly in segmentation:
            
            # Convert the polygon into a mask
            poly_points = np.array(poly).reshape(-1, 2)
            img = Image.new('L', (image.width, image.height), 0)  # Create a grayscale image
            ImageDraw.Draw(img).polygon([tuple(point) for point in poly_points], outline=1, fill=1)
            binary_mask = np.maximum(binary_mask, np.array(img))

    
    
    elif isinstance(segmentation, dict) and 'counts' in segmentation:
        height, width = image.height, image.width
        rle = segmentation['counts']
        
        if isinstance(rle, list):  
            rle_mask = decode_rle(rle, height, width)
            binary_mask = np.maximum(binary_mask, rle_mask)

# Save the binary mask as a PNG file (grayscale)
mask_image = Image.fromarray(binary_mask * 255)  # Convert mask from 0/1 to 0/255
mask_image.save("binary_mask.png")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Imagem Original")

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title("Máscara Binária Gerada")

plt.show()
