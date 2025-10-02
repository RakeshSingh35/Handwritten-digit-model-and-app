import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

# ------------------------------
# Digit Recognizer App (Tkinter)
# ------------------------------

class DigitRecognizerApp:
    def __init__(self, model):
        self.model = model

        self.window = tk.Tk()
        self.window.title("Handwritten Digit Recognition")

        self.canvas = tk.Canvas(self.window, width=280, height=280, bg='white')
        self.canvas.grid(row=0, column=0, columnspan=2)
        
        self.button_predict = tk.Button(self.window, text="Predict", command=self.predict_digit)
        self.button_predict.grid(row=1, column=0)
        
        self.button_clear = tk.Button(self.window, text="Clear", command=self.clear_canvas)
        self.button_clear.grid(row=1, column=1)
        
        self.label = tk.Label(self.window, text="Draw a digit and click Predict", font=('Helvetica', 14))
        self.label.grid(row=2, column=0, columnspan=2)
        
        # For drawing
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (280, 280), 'white')  # PIL image
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        r = 8  # Brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
        self.draw_image.ellipse([x-r, y-r, x+r, y+r], fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0,0,280,280], fill='white')
        self.label.config(text="Draw a digit and click Predict")

    def preprocess_image(self):
        # Convert to numpy array
        img = np.array(self.image)

        # Invert (white background, black digit)
        img = ImageOps.invert(Image.fromarray(img))
        img_array = np.array(img)

        # Find bounding box of non-white pixels (digit area)
        coords = np.argwhere(img_array > 0)
        if coords.size == 0:
            return None

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop to digit region
        img_cropped = img_array[y_min:y_max+1, x_min:x_max+1]

        # Resize keeping aspect ratio + padding to center (like MNIST)
        img_pil = Image.fromarray(img_cropped)
        max_side = max(img_cropped.shape)
        new_img = Image.new("L", (max_side, max_side), 0)  # black background
        paste_x = (max_side - img_cropped.shape[1]) // 2
        paste_y = (max_side - img_cropped.shape[0]) // 2
        new_img.paste(img_pil, (paste_x, paste_y))

        # Final resize to 28x28
        img_resized = new_img.resize((28, 28), Image.LANCZOS)

        # Normalize
        img_resized = np.array(img_resized).astype('float32') / 255.0
        img_resized = img_resized.reshape(1, 28, 28, 1)

        return img_resized

    def predict_digit(self):
        img_resized = self.preprocess_image()
        if img_resized is None:
            self.label.config(text="No digit detected. Draw again!")
            return

        # Predict digit
        pred = self.model.predict(img_resized)
        digit = np.argmax(pred)
        confidence = np.max(pred)

        self.label.config(text=f"Prediction: {digit} (Confidence: {confidence:.2f})")

    def run(self):
        self.window.mainloop()

# ------------------------------
# Main Function
# ------------------------------
def main():
    """Main function to run the digit recognition app"""
    try:
        # Load the trained model
        print("Loading model...")
        model = tf.keras.models.load_model('mnist_cnn_model.keras')
        
        # Create and run the app
        print("Starting GUI application...")
        app = DigitRecognizerApp(model)
        app.run()
    except FileNotFoundError:
        print("Error: Model file 'mnist_cnn_model.keras' not found!")
        print("Please make sure the model file exists in the same directory.")
        input("Press Enter to exit...")
    except Exception as e:
        print(f"An error occurred: {e}")
        input("Press Enter to exit...")

# ------------------------------
# This ensures the script runs only when executed directly
# ------------------------------
if __name__ == "__main__":
    main()