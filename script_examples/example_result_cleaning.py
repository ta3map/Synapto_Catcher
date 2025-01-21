import sys, os
import tkinter as tk

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from graphical_processor import FileDeletionDialog

# Example usage of the dialog
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # List of folders to search for files
    folders_to_search = [
        r"E:\Tile Photos\240812 AG\Experiment-1538_results"
    ]

    # Create the file deletion dialog
    dialog = FileDeletionDialog(folders_to_search)
    dialog.mainloop()
