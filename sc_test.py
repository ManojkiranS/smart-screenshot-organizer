from PIL import Image
import pytesseract
image_path = "test_image.png" 

try:
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    print("Start")
    print(text)
    print("End") 
except FileNotFoundError:
    print(f"Could not find the file '{image_path}' ")
except Exception as e:
    print(f"An error occurred: {e}")