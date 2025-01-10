import tkinter as tk
from tkinter import scrolledtext
import sys
import re

class RedirectText(object):
    def __init__(self, text_widget):
        self.text_widget = text_widget
        # Определяем стили для хэштегов
        self.text_widget.tag_configure("bold", font=("Helvetica", 12, "bold"))
        self.text_widget.tag_configure("red", foreground="red")
        self.text_widget.tag_configure("large", font=("Helvetica", 16))
        self.text_widget.tag_configure("italic", font=("Helvetica", 12, "italic"))

    def write(self, string):
        # Здесь мы ищем хэштеги, которые не будут отображаться, но будут влиять на стиль текста
        hashtags = re.findall(r'#\w+', string)

        # Если хэштеги найдены, применяем их, но не выводим
        if hashtags:
            self.apply_style(hashtags)
        else:
            # Вставляем текст без изменений, если хэштегов нет
            self.text_widget.insert(tk.END, string)
            self.text_widget.see(tk.END)  # Автопрокрутка

    def apply_style(self, hashtags):
        # Получаем весь текст
        start = "1.0"
        end = tk.END

        # Удаляем все стили перед применением новых
        self.text_widget.tag_remove("bold", start, end)
        self.text_widget.tag_remove("red", start, end)
        self.text_widget.tag_remove("large", start, end)
        self.text_widget.tag_remove("italic", start, end)

        # Применяем стили по найденным хэштегам
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

        # Создаем текстовое поле с прокруткой
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=15)
        self.text_area.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Перенаправляем print в текстовое поле
        redirect_text = RedirectText(self.text_area)
        sys.stdout = redirect_text

        # Изначально выводим тестовое сообщение
        print("TEST\n")

        # Счетчик для изменения стилей
        self.style_counter = 0

        # Добавляем кнопку для изменения стиля текста
        self.style_button = tk.Button(self.root, text="Change Style", command=self.on_button_click)
        self.style_button.pack(pady=10)

    def on_button_click(self):
        # Здесь будет происходить вывод хэштегов через print для изменения стиля
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

        # Увеличиваем счётчик для циклического изменения стиля
        self.style_counter = (self.style_counter + 1) % 6

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()
