import json
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt
from mask_generator import create_binary_mask



with open('./selected-segmentation/result.json', 'r') as f:
    coco_data = json.load(f)
    

for img in coco_data['images']:
    path = './selected-segmentation/' + img['file_name']
    create_binary_mask(coco_data, path, img['id'] )
    
    
