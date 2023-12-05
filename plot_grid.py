from PIL import Image, ImageDraw

def plot_grid(image_path, grid_number):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    width, height = img.size

    # Calculate step size for each grid
    step_x = width / grid_number
    step_y = height / grid_number

    # Draw vertical lines
    for i in range(1, grid_number):
        draw.line((step_x * i, 0, step_x * i, height), fill='red', width=1)

    # Draw horizontal lines
    for j in range(1, grid_number):
        draw.line((0, step_y * j, width, step_y * j), fill='red', width=1)
    img.save('./tmp.jpg')
plot_grid('./dataset/image_101.jpg',64)