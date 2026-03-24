import cv2
import numpy as np

# Stałe i konfiguracja
RECTANGLE_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
CIRCLE_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

TXT_INITIAL_POSITION = (10, 40)
TXT_Y_OFFSET_FACTOR = 4 / 100
TXT_X_OFFSET_FACTOR = 80 / 100

TXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TXT_SCALE = 0.6
TXT_COLOR = (200, 120, 200)
TXT_THICKNESS = 2
TXT_LINE_TYPE = 1


# Główna logika
def count_coins(image_data):
    """
    Dla jednego obrazu:
    - wykrywa tacę,
    - wykrywa monety,
    - dzieli monety na te na tacy i poza tacą,
    - oblicza wartości,
    - wypisuje wynik na ekran i na obrazie.
    """
    color_img, gray_img = image_data

    tray_points = detect_rectangle(gray_img)
    left_border, top_border, right_border, bottom_border = tray_points

    circles = detect_circles(gray_img)

    money_on_tray = []
    money_outside_tray = []

    if circles is not None and len(circles[0]) > 0:
        max_radius = max(circle[2] for circle in circles[0])

        for x, y, r in circles[0]:
            # moneta na tacy
            if left_border < x < right_border and bottom_border < y < top_border:
                money_on_tray.append(classify_coin_by_radius(r, max_radius))
            # monet poza taca
            else:
                money_outside_tray.append(classify_coin_by_radius(r, max_radius))

    money_on_tray_float = [money/100 for money in money_on_tray]
    money_outside_tray_float = [money/100 for money in money_outside_tray]
    print("\n" * 5)
    print(f"pieniadze na tacy [{len(money_on_tray)}]:   {money_on_tray_float}   =   {sum(money_on_tray)/100:.2f} PLN")
    print(f"pieniadze poza taca [{len(money_outside_tray)}]: {money_outside_tray_float}   =   {sum(money_outside_tray)/100:.2f} PLN")

    y_offset = int(color_img.shape[0] * TXT_Y_OFFSET_FACTOR)
    x_offset = int(color_img.shape[1] * TXT_X_OFFSET_FACTOR)

    draw_text_on_image(color_img, f"na tacy [{len(money_on_tray)}]: {money_on_tray_float}")
    draw_text_on_image(color_img, f"poza taca [{len(money_outside_tray)}]: {money_outside_tray_float}", (0, y_offset))

    draw_text_on_image(color_img, f"= {sum(money_on_tray)/100:.2f} PLN", (x_offset, 0))
    draw_text_on_image(color_img, f"= {sum(money_outside_tray)/100:.2f} PLN", (x_offset, y_offset))

    cv2.imshow("img", color_img)
    cv2.waitKey()


# Rysowanie tekstu
def draw_text_on_image(img, text, position_offset=(0, 0)):
    """
    Rysuje tekst na obrazie w zadanej pozycji.
    """
    x = TXT_INITIAL_POSITION[0] + position_offset[0]
    y = TXT_INITIAL_POSITION[1] + position_offset[1]

    cv2.putText(
        img,
        text,
        (x, y),
        TXT_FONT,
        TXT_SCALE,
        TXT_COLOR,
        TXT_THICKNESS,
        TXT_LINE_TYPE
    )


# Klasyfikacja monety
def classify_coin_by_radius(radius, max_radius):
    """
    Rozróżnia monetę 5 zł i 5 gr na podstawie promienia.
    Największe promienie są traktowane jako 5 zł.
    """
    threshold = max_radius * 90 / 100

    if radius >= threshold:
        return int(500)
    return int(5)


# Wykrywanie tacy
def detect_rectangle(gray_img):
    """
    Wykrywa przybliżony prostokąt tacy na podstawie HoughLinesP.
    Zwraca punkty: [left, top, right, bottom].
    """
    processed = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, RECTANGLE_MORPH_KERNEL, iterations=1)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, RECTANGLE_MORPH_KERNEL, iterations=1)
    processed = cv2.Canny(processed, 80, 250, None, 3)

    view = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(
        processed,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        lines=None,
        minLineLength=50,
        maxLineGap=150
    )

    # [left, top, right, bottom]
    rectangle_points = [processed.shape[0], 0, 0, processed.shape[1]]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            rectangle_points[0] = min(rectangle_points[0], x1, x2)  # left
            rectangle_points[1] = max(rectangle_points[1], y1, y2)  # top
            rectangle_points[2] = max(rectangle_points[2], x1, x2)  # right
            rectangle_points[3] = min(rectangle_points[3], y1, y2)  # bottom

    left, top, right, bottom = rectangle_points

    p1 = (left, top)
    p2 = (right, top)
    p3 = (left, bottom)
    p4 = (right, bottom)

    cv2.line(view, p1, p2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(view, p3, p4, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(view, p1, p3, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(view, p2, p4, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("detect tray", view)

    return rectangle_points


# Wykrywanie monet
def detect_circles(gray_img):
    """
    Wykrywa monety za pomocą HoughCircles.
    Zwraca tablicę okręgów w formacie jak z OpenCV:
    circles[0] -> [(x, y, r), ...]
    """
    processed = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, CIRCLE_MORPH_KERNEL, iterations=3)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, CIRCLE_MORPH_KERNEL, iterations=1)
    processed = cv2.medianBlur(processed, 5)

    view = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(
        processed,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=40
    )

    if circles is None:
        cv2.imshow("detect money", view)
        return None

    detected_circles = np.uint16(np.around(circles))

    for x, y, r in detected_circles[0]:
        cv2.circle(view, (x, y), r, (0, 255, 0), 3)
        cv2.circle(view, (x, y), 2, (0, 255, 255), 3)

    cv2.imshow("detect money", view)

    return detected_circles


# Main
def main() -> int:
    """
    Wczytuje obrazy tray1.jpg ... tray8.jpg
    i przetwarza je jeden po drugim.
    """
    images = []  # [(color_img, gray_img), ...]

    for i in range(1, 9):
        img = cv2.imread(f"tray{i}.jpg")

        if img is None:
            print(f"Nie udalo sie wczytac obrazu: tray{i}.jpg")
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append((img, gray_img))

    if len(images) == 0:
        print("Error reading images!")
        return 1

    for image_data in images:
        count_coins(image_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
