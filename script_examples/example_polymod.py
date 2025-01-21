import cv2
import numpy as np

# Координаты полигона
blob = np.array([[150, 100], [300, 80], [400, 200], [350, 400], [200, 350], [100, 300], [50, 150]], dtype=np.float32)
blob = blob.astype(np.int32)

selected_vertex = None
dragging = False
tolerance = 10  # Радиус для определения клика на вершину
window_name = "Polygon Editor"

def distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def is_on_edge(point, poly, tolerance=5):
    # Определение, попадает ли клик на ребро полигона
    for i in range(len(poly)):
        next_point = poly[(i + 1) % len(poly)]
        d = np.cross(next_point - poly[i], point - poly[i]) / distance(poly[i], next_point)
        if abs(d) < tolerance:
            # Если клик на ребре, возвращаем индекс ребра и ближайшую точку на ребре
            vec = next_point - poly[i]
            t = np.dot(point - poly[i], vec) / np.dot(vec, vec)
            if 0 <= t <= 1:
                nearest_point = poly[i] + t * vec
                return i, nearest_point.astype(np.int32)
    return None, None

def draw_polygon(img, points):
    # Отрисовка полигона и его вершин
    img = cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
    for point in points:
        cv2.circle(img, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)
    return img

def mouse_callback(event, x, y, flags, param):
    global selected_vertex, dragging, blob

    if event == cv2.EVENT_LBUTTONDOWN:
        # Проверка, кликнул ли пользователь на вершину
        for i, point in enumerate(blob):
            if distance(point, np.array([x, y])) < tolerance:
                selected_vertex = i
                dragging = True
                return
        
        # Если пользователь не кликнул на вершину, проверяем ребра
        edge_index, new_vertex = is_on_edge(np.array([x, y]), blob)
        if edge_index is not None:
            # Вставляем новую вершину в место клика
            next_index = (edge_index + 1) % len(blob)
            blob = np.insert(blob, next_index, new_vertex, axis=0)

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging and selected_vertex is not None:
            # Перемещаем вершину
            blob[selected_vertex] = [x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        # Прекращаем перемещение
        dragging = False
        selected_vertex = None

# Создаем окно и задаем коллбэк для мыши
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:
    img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    img = draw_polygon(img, blob)

    cv2.imshow(window_name, img)
    
    # Выход по клавише ESC
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
