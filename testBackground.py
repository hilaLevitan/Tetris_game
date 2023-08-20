# import cv2
# import numpy as np
# background = cv2.imread('./pics/Canvas.png')
# overlay = cv2.imread('./pics/tet/blue.png', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
#
# scale_percent = 50  # percent of original size
# width = int(overlay.shape[1] * scale_percent / 100)
# height = int(overlay.shape[0] * scale_percent / 100)
# dim = (width, height)
#
# # resize image
# resized = cv2.resize(overlay, dim, interpolation=cv2.INTER_AREA)
#
# height, width = resized.shape[:2]
# for y in range(height):
#     for x in range(width):
#         overlay_color = resized[y, x, :3]  # first three elements are color (RGB)
#         overlay_alpha = resized[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0
#
#         # get the color from the background image
#         background_color = background[y, x]
#
#         # combine the background color and the overlay color weighted by alpha
#         composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha
#
#         # update the background image in place
#         background[y, x] = composite_color
# cv2.imshow('win',background)
#
#
# # cv2.imwrite('combined.png', background)
# cv2.waitKey(12000)

# import cv2
#
# # Load the image
# image = cv2.imread("./pics/tetresized/pink.png", cv2.IMREAD_COLOR)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply thresholding to create a binary mask
# _, binary_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#
# # Show the original image and the binary mask
# cv2.imshow("Original Image", image)
# cv2.imshow("Binary Mask", binary_mask)
#
# # Wait for a key press to close the windows
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

# Load the image
image = cv2.imread("./pics/Canvas.png")

# Function to add text to the image
def add_text_to_image(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White color
    thickness = 2

    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image.shape[1] - text_size[0]) // 2  # Centered horizontally
    text_y = image.shape[0] - 30  # Some distance from the bottom
    text_position = (text_x, text_y)

    cv2.putText(image, text, text_position, font, font_scale, font_color, thickness)
    return image

# Define the text to add
text_to_add = "Hello, OpenCV!"

# Add text to the image
image_with_text = add_text_to_image(image.copy(), text_to_add)

# Display the image with text
cv2.imshow("Image with Text", image_with_text)
cv2.waitKey(0)

# Define the region containing the text (replace with your actual values)
x1, y1, x2, y2 = 100, 200, 400, 300

# Create a mask for the text region
mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

# Inpainting to remove the text region
inpaint_image = cv2.inpaint(image_with_text, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Display the image with text removed
cv2.imshow("Image with Text Removed", inpaint_image)
cv2.waitKey(0)
cv2.destroyAllWindows()