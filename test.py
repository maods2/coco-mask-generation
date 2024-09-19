from PIL import Image, ImageDraw

image_path = 'images/7297f39a-20240911_080133.jpg'
image = Image.open(image_path)

print((image.height, image.width))

image_path = 'binary_mask.png'
image = Image.open(image_path)

print((image.height, image.width))