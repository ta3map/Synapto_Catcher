import sys, os

# Add the current directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from graphical_processor import create_excel_snapshot_to_image

# Пример использования
filepath = "E:\iMAGES\protocol_test.xlsx"  # Замените на реальный путь к файлу
image = create_excel_snapshot_to_image(filepath)
image.show()
