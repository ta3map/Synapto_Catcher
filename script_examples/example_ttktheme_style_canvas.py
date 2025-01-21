import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk

class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Пример приложения")
        
        # Создаем виджеты
        self.create_widgets()

    def create_widgets(self):
        # Пример с Canvas
        self.canvas = tk.Canvas(self.root, width=200, height=200)
        self.canvas.pack(padx=10, pady=10)

        # Пример с Frame
        self.frame = ttk.Frame(self.root, width=200, height=200)
        self.frame.pack(padx=10, pady=10)

        # Пример с обычным Label
        self.label = tk.Label(self.root, text="Это обычный Label")
        self.label.pack(pady=10)

# Функция для применения всех параметров темы
def apply_theme(root, style):
    # Получаем цвет фона и шрифт из текущей темы
    theme_background = style.lookup('TFrame', 'background')
    theme_font = style.lookup('TLabel', 'font')
    theme_foreground = style.lookup('TLabel', 'foreground')

    # Устанавливаем фон для главного окна
    root.configure(bg=theme_background)

    # Рекурсивная функция для применения фона, шрифта и цвета текста ко всем виджетам
    def update_widget_appearance(widget):
        try:
            widget.configure(bg=theme_background)
        except tk.TclError:
            pass  # Игнорируем виджеты, которые не поддерживают изменение фона

        try:
            widget.configure(font=theme_font, fg=theme_foreground)
        except (tk.TclError, AttributeError):
            pass  # Игнорируем виджеты, которые не поддерживают изменение шрифта или цвета текста

        for child in widget.winfo_children():
            update_widget_appearance(child)

    # Применяем изменения ко всем виджетам
    update_widget_appearance(root)

    # Попробуем изменить цвет заголовка окна (шапки)
    try:
        root.wm_attributes("-titlepath", theme_background)
    except tk.TclError:
        pass  # Изменение цвета заголовка не поддерживается на всех системах

if __name__ == "__main__":
    # Создаем окно с темой
    root = ThemedTk(theme="black")

    # Инициализируем объект приложения
    app = MyApp(root)

    # Создаем объект стиля для работы с темой
    style = ttk.Style()

    # Применяем все аспекты темы снаружи объекта
    apply_theme(root, style)

    # Запускаем главный цикл
    root.mainloop()
