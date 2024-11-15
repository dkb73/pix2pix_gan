from PIL import Image
import colorgram

# Load the image
image_path = 'img1.png'
img = Image.open(image_path)
width, height = img.size

print(f"Image size: {width}x{height}")  

# Define regions
regions = {
    "Top-Left": (0, 0, width // 2, height // 2),
    "Bottom-Left": (0, height // 2, width // 2, height),
    "Top-Right": (width // 2, 0, width, height // 2),
    "Bottom-Right": (width // 2, height // 2, width, height),
}

# Process each region
for region_name, box in regions.items():
    # Crop the region
    cropped = img.crop(box)
    
    # Save cropped region temporarily (colorgram works on image files)
    cropped.save('temp.png')
    
    # Extract colors from the cropped region
    colors = colorgram.extract('temp.png', 5)  # Extract top 5 colors
    
    print(f"Colors in {region_name} region:")
    for color in colors:
        rgb = color.rgb
        proportion = color.proportion
        print(f"  RGB: {rgb}, Proportion: {proportion:.2f}")

