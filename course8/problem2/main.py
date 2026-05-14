import cv2


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f'이미지를 찾을 수 없습니다: {image_path}')

    return image


def task_1_flip_rotate(image_path):
    image = load_image(image_path)

    show_image('Original Image', image)

    flipped_vertical = cv2.flip(image, 0)
    show_image('Vertical Flip', flipped_vertical)

    flipped_horizontal = cv2.flip(image, 1)
    show_image('Horizontal Flip', flipped_horizontal)

    rotated_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    show_image('Rotate 90 Clockwise', rotated_90)

    rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
    show_image('Rotate 180', rotated_180)

    upsampled = cv2.pyrUp(image)
    show_image('Upsampled 2x', upsampled)


def task_2_resize_scaling_crop(image_path):
    image = load_image(image_path)

    resized_640_480 = cv2.resize(image, (640, 480))
    show_image('Resize 640x480', resized_640_480)

    resized_1024_768 = cv2.resize(image, (1024, 768))
    show_image('Resize 1024x768', resized_1024_768)

    scaled_image = cv2.resize(
        image,
        None,
        fx=0.3,
        fy=0.7,
        interpolation=cv2.INTER_AREA
    )
    show_image('Scaled fx 0.3 fy 0.7', scaled_image)

    crop_image = image[100:300, 150:400].copy()
    show_image('Deep Copied Crop Image', crop_image)


def task_2_bonus_crop_people(image_path):
    image = load_image(image_path)

    person_1 = image[50:350, 30:180].copy()
    person_2 = image[60:360, 200:360].copy()
    person_3 = image[70:370, 390:540].copy()

    show_image('Person 1', person_1)
    show_image('Person 2', person_2)
    show_image('Person 3', person_3)


def task_3_color_inverse(image_path):
    image = load_image(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image('Gray Image', gray_image)

    inverse_image = 255 - image
    show_image('Inverse Image', inverse_image)


def task_3_bonus_histogram(image_path):
    image = load_image(image_path)
    inverse_image = 255 - image

    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_inverse = cv2.cvtColor(inverse_image, cv2.COLOR_BGR2GRAY)

    original_hist = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
    inverse_hist = cv2.calcHist([gray_inverse], [0], None, [256], [0, 256])

    hist_width = 512
    hist_height = 400

    original_hist_image = create_histogram_image(
        original_hist,
        hist_width,
        hist_height
    )
    inverse_hist_image = create_histogram_image(
        inverse_hist,
        hist_width,
        hist_height
    )

    show_image('Original Histogram', original_hist_image)
    show_image('Inverse Histogram', inverse_hist_image)


def create_histogram_image(histogram, width, height):
    histogram_image = 255 * cv2.UMat(height, width, cv2.CV_8UC3).get()
    cv2.normalize(histogram, histogram, 0, height, cv2.NORM_MINMAX)

    bin_width = int(width / 256)

    for index in range(1, 256):
        x_1 = bin_width * (index - 1)
        y_1 = height - int(histogram[index - 1][0])
        x_2 = bin_width * index
        y_2 = height - int(histogram[index][0])

        cv2.line(
            histogram_image,
            (x_1, y_1),
            (x_2, y_2),
            (0, 0, 0),
            1
        )

    return histogram_image


def task_4_binary_edge_blur(image_path, blur_image_path):
    image = load_image(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(
        gray_image,
        127,
        255,
        cv2.THRESH_BINARY
    )
    show_image('Binary Image', binary_image)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.convertScaleAbs(sobel_x + sobel_y)
    show_image('Sobel Edge', sobel)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    show_image('Laplacian Edge', laplacian)

    canny = cv2.Canny(gray_image, 100, 200)
    show_image('Canny Edge', canny)

    blur_image = load_image(blur_image_path)

    blurred_image = cv2.GaussianBlur(blur_image, (15, 15), 0)
    show_image('Blurred Image', blurred_image)


def task_4_bonus_partial_blur(image_path):
    image = load_image(image_path)

    result_image = image.copy()

    target_area = result_image[100:300, 150:400]
    blurred_area = cv2.GaussianBlur(target_area, (31, 31), 0)

    result_image[100:300, 150:400] = blurred_area

    show_image('Partial Blur Image', result_image)


def task_5_hsv_channels(image_path):
    image = load_image(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h_channel, s_channel, v_channel = cv2.split(hsv_image)

    show_image('H Channel', h_channel)
    show_image('S Channel', s_channel)
    show_image('V Channel', v_channel)


def task_5_bonus_bgr_channels(image_path):
    image = load_image(image_path)

    blue_channel, green_channel, red_channel = cv2.split(image)

    show_image('Blue Channel', blue_channel)
    show_image('Green Channel', green_channel)
    show_image('Red Channel', red_channel)


def task_6_object_labeling(image_path):
    image = load_image(image_path)
    result_image = image.copy()

    objects = [
        {
            'name': 'Object 1',
            'box': (50, 80, 180, 220),
            'text_position': (40, 50)
        },
        {
            'name': 'Object 2',
            'box': (250, 100, 420, 260),
            'text_position': (240, 70)
        },
        {
            'name': 'Object 3',
            'box': (470, 120, 620, 300),
            'text_position': (460, 90)
        }
    ]

    red_color = (0, 0, 255)

    for item in objects:
        x_1, y_1, x_2, y_2 = item['box']
        text_x, text_y = item['text_position']

        cv2.rectangle(
            result_image,
            (x_1, y_1),
            (x_2, y_2),
            red_color,
            2
        )

        cv2.putText(
            result_image,
            item['name'],
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            red_color,
            2
        )

        cv2.line(
            result_image,
            (text_x, text_y + 10),
            (x_1, y_1),
            red_color,
            2
        )

    show_image('Object Labeling', result_image)


def task_6_bonus_shape_labeling(image_path):
    image = load_image(image_path)
    result_image = image.copy()

    red_color = (0, 0, 255)

    cv2.rectangle(
        result_image,
        (50, 80),
        (180, 220),
        red_color,
        2
    )
    cv2.putText(
        result_image,
        'Box Object',
        (40, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        red_color,
        2
    )

    cv2.circle(
        result_image,
        (330, 180),
        70,
        red_color,
        2
    )
    cv2.putText(
        result_image,
        'Circle Object',
        (260, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        red_color,
        2
    )

    triangle_points = cv2.UMat(
        [[[520, 90], [450, 250], [590, 250]]]
    ).get()

    cv2.polylines(
        result_image,
        triangle_points,
        True,
        red_color,
        2
    )
    cv2.putText(
        result_image,
        'Triangle Object',
        (430, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        red_color,
        2
    )

    show_image('Shape Labeling', result_image)


def main():
    sample_image_path = 'images/apple.jpg'
    people_image_path = 'images/people.jpg'
    blur_image_path = 'images/blur_sample.jpg'
    objects_image_path = 'images/objects.jpg'

    task_1_flip_rotate(sample_image_path)
    task_2_resize_scaling_crop(sample_image_path)
    task_2_bonus_crop_people(people_image_path)
    task_3_color_inverse(sample_image_path)
    task_3_bonus_histogram(sample_image_path)
    task_4_binary_edge_blur(sample_image_path, blur_image_path)
    task_4_bonus_partial_blur(blur_image_path)
    task_5_hsv_channels(sample_image_path)
    task_5_bonus_bgr_channels(sample_image_path)
    task_6_object_labeling(objects_image_path)
    task_6_bonus_shape_labeling(objects_image_path)


if __name__ == '__main__':
    main()