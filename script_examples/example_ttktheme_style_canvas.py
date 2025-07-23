import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Пример приложения")
        
        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        # Example with Canvas
        self.canvas = tk.Canvas(self.root, width=200, height=200)
        self.canvas.pack(padx=10, pady=10)

        # Example with Frame
        self.frame = ttk.Frame(self.root, width=200, height=200)
        self.frame.pack(padx=10, pady=10)

        # Example with regular Label
        self.label = tk.Label(self.root, text="Это обычный Label")
        self.label.pack(pady=10)

    # Function to apply all theme parameters
def apply_theme(root, style):
            # Get background color and font from current theme
    theme_background = style.lookup('TFrame', 'background')
    theme_font = style.lookup('TLabel', 'font')
    theme_foreground = style.lookup('TLabel', 'foreground')

            # Set background for main window
    root.configure(bg=theme_background)

            # Recursive function to apply background, font and text color to all widgets
    def update_widget_appearance(widget):
        try:
            widget.configure(bg=theme_background)
        except tk.TclError:
            pass  # Ignore widgets that don't support background change

        try:
            widget.configure(font=theme_font, fg=theme_foreground)
        except (tk.TclError, AttributeError):
            pass  # Ignore widgets that don't support font or text color change

        for child in widget.winfo_children():
            update_widget_appearance(child)

            # Apply changes to all widgets
    update_widget_appearance(root)

            # Try to change window title bar color
    try:
        root.wm_attributes("-titlepath", theme_background)
    except tk.TclError:
        pass  # Title bar color change is not supported on all systems

if __name__ == "__main__":
    # Create window with theme
    root = ThemedTk(theme="black")

    # Initialize application object
    app = MyApp(root)

    # Create style object to work with theme
    style = ttk.Style()

    # Apply all aspects of theme outside object
    apply_theme(root, style)

    # Start main loop
    root.mainloop()
