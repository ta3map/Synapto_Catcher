import sys, os

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from graphical_processor import create_excel_snapshot_to_image

# Usage example
filepath = "E:\iMAGES\protocol_test.xlsx"  # Replace with real file path
image = create_excel_snapshot_to_image(filepath)
image.show()
