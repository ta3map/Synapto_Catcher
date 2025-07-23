import tkinter as tk
from tkinter import scrolledtext
import sys
import re

class RedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        # Define styles for hashtags
        self.text_widget.tag_configure("bold", font=("Helvetica", 12, "bold"))
        self.text_widget.tag_configure("red", foreground="red")
        self.text_widget.tag_configure("large", font=("Helvetica", 16))
        self.text_widget.tag_configure("italic", font=("Helvetica", 12, "italic"))

    def write(self, string):
        # Here we search for hashtags that won't be displayed but will affect text style
        hashtags = re.findall(r'#\w+', string)

        # If hashtags are found, apply them but don't display
        if hashtags:
            self.apply_style(hashtags)
        else:
            # Insert text unchanged if no hashtags
            self.text_widget.insert(tk.END, string)
            self.text_widget.see(tk.END)  # Auto-scroll

    def apply_style(self, hashtags):
        # Get all text
        start = "1.0"
        end = tk.END

        # Remove all styles before applying new ones
        self.text_widget.tag_remove("bold", start, end)
        self.text_widget.tag_remove("red", start, end)
        self.text_widget.tag_remove("large", start, end)
        self.text_widget.tag_remove("italic", start, end)

        # Apply styles based on found hashtags
        if "#bold" in hashtags:
            self.text_widget.tag_add("bold", start, end)
        if "#red" in hashtags:
            self.text_widget.tag_add("red", start, end)
        if "#large" in hashtags:
            self.text_widget.tag_add("large", start, end)
        if "#italic" in hashtags:
            self.text_widget.tag_add("italic", start, end)

    def flush(self):
        pass


class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Redirected Output with Hashtags")

        # Create text field with scrolling
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=15)
        self.text_area.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Redirect print to text field
        redirect_text = RedirectText(self.text_area)
        sys.stdout = redirect_text

        # Initially output test message
        print("TEST\n")

        # Counter for changing styles
        self.style_counter = 0

        # Add button to change text style
        self.style_button = tk.Button(self.root, text="Change Style", command=self.on_button_click)
        self.style_button.pack(pady=10)

    def on_button_click(self):
        # Here hashtags will be output via print to change style
        if self.style_counter == 0:
            print("#bold")
        elif self.style_counter == 1:
            print("#red")
        elif self.style_counter == 2:
            print("#large")
        elif self.style_counter == 3:
            print("#italic")
        elif self.style_counter == 4:
            print("#bold #red")
        elif self.style_counter == 5:
            print("#large #italic")

        # Increment counter for cyclic style change
        self.style_counter = (self.style_counter + 1) % 6

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()
