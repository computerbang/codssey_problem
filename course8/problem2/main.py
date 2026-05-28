import cv2
from datetime import datetime


WAIT_TIME = 0


def show_image(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(WAIT_TIME)
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

    height, width = image.shape[:2]

    start_x = width // 4
    end_x = start_x + width // 2
    start_y = height // 4
    end_y = start_y + height // 2

    cropped_image = image[start_y:end_y, start_x:end_x].copy()
    show_image('Deep Copied Crop Image', cropped_image)


def task_3_color_inverse(image_path):
    image = load_image(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image('Gray Image', gray_image)

    inverse_image = 255 - image
    show_image('Inverse Image', inverse_image)


def create_histogram_image(histogram, width, height):
    histogram_image = cv2.UMat(height, width, cv2.CV_8UC3).get()

    cv2.normalize(
        histogram,
        histogram,
        0,
        height,
        cv2.NORM_MINMAX
    )

    bin_width = width // 256

    for index in range(1, 256):
        x_1 = bin_width * (index - 1)
        y_1 = height - int(histogram[index - 1][0])
        x_2 = bin_width * index
        y_2 = height - int(histogram[index][0])

        cv2.line(
            histogram_image,
            (x_1, y_1),
            (x_2, y_2),
            (255, 255, 255),
            1
        )

    return histogram_image


def task_3_bonus_histogram(image_path):
    image = load_image(image_path)

    inverse_image = 255 - image

    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_inverse = cv2.cvtColor(inverse_image, cv2.COLOR_BGR2GRAY)

    original_histogram = cv2.calcHist(
        [gray_original],
        [0],
        None,
        [256],
        [0, 256]
    )

    inverse_histogram = cv2.calcHist(
        [gray_inverse],
        [0],
        None,
        [256],
        [0, 256]
    )

    original_histogram_image = create_histogram_image(
        original_histogram,
        512,
        400
    )

    inverse_histogram_image = create_histogram_image(
        inverse_histogram,
        512,
        400
    )

    show_image('Original Histogram', original_histogram_image)
    show_image('Inverse Histogram', inverse_histogram_image)


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

    sobel_x = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel = cv2.addWeighted(
        cv2.convertScaleAbs(sobel_x),
        0.5,
        cv2.convertScaleAbs(sobel_y),
        0.5,
        0
    )
    show_image('Sobel Edge', sobel)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    show_image('Laplacian Edge', laplacian)

    canny = cv2.Canny(gray_image, 100, 200)
    show_image('Canny Edge', canny)

    blur_image = load_image(blur_image_path)

    blurred_image = cv2.GaussianBlur(
        blur_image,
        (15, 15),
        0
    )
    show_image('Blurred Image', blurred_image)


def task_4_bonus_partial_blur(image_path):
    image = load_image(image_path)
    result_image = image.copy()

    height, width = result_image.shape[:2]

    start_x = width // 4
    end_x = start_x + width // 2
    start_y = height // 4
    end_y = start_y + height // 2

    target_area = result_image[start_y:end_y, start_x:end_x]
    blurred_area = cv2.GaussianBlur(
        target_area,
        (31, 31),
        0
    )

    result_image[start_y:end_y, start_x:end_x] = blurred_area

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

    height, width = result_image.shape[:2]

    red_color = (0, 0, 255)

    box_start = (
        width // 4,
        height // 4
    )

    box_end = (
        width * 3 // 4,
        height * 3 // 4
    )

    text_position = (
        width // 4,
        height // 4 - 30
    )

    if text_position[1] < 30:
        text_position = (
            width // 4,
            30
        )

    cv2.rectangle(
        result_image,
        box_start,
        box_end,
        red_color,
        2
    )

    cv2.putText(
        result_image,
        'Main Object',
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        red_color,
        2
    )

    cv2.line(
        result_image,
        (text_position[0], text_position[1] + 10),
        box_start,
        red_color,
        2
    )

    show_image('Object Labeling', result_image)


def task_6_bonus_shape_labeling(image_path):
    image = load_image(image_path)
    result_image = image.copy()

    height, width = result_image.shape[:2]

    red_color = (0, 0, 255)

    cv2.rectangle(
        result_image,
        (width // 10, height // 5),
        (width // 3, height // 2),
        red_color,
        2
    )

    cv2.putText(
        result_image,
        'Rectangle',
        (width // 10, height // 5 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        red_color,
        2
    )

    cv2.circle(
        result_image,
        (width // 2, height // 2),
        min(width, height) // 8,
        red_color,
        2
    )

    cv2.putText(
        result_image,
        'Circle',
        (width // 2 - 40, height // 2 - 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        red_color,
        2
    )

    point_1 = (
        width * 3 // 4,
        height // 4
    )

    point_2 = (
        width * 2 // 3,
        height * 3 // 4
    )

    point_3 = (
        width * 5 // 6,
        height * 3 // 4
    )

    cv2.line(result_image, point_1, point_2, red_color, 2)
    cv2.line(result_image, point_2, point_3, red_color, 2)
    cv2.line(result_image, point_3, point_1, red_color, 2)

    cv2.putText(
        result_image,
        'Triangle',
        (width * 2 // 3, height // 4 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        red_color,
        2
    )

    show_image('Shape Labeling', result_image)


def print_task_title(title):
    print()
    print('=' * 50)
    print(title)
    print('=' * 50)


def main():
    apple_image_path = 'images/apple.jpg'
    orange_image_path = 'images/orange.jpg'

    print_task_title('1. 이미지 반전 및 회전')
    task_1_flip_rotate(apple_image_path)

    print_task_title('2. 이미지 리사이즈, 스케일링, 크롭')
    task_2_resize_scaling_crop(apple_image_path)

    print_task_title('3. 색상 변환과 역상 처리')
    task_3_color_inverse(apple_image_path)
    task_3_bonus_histogram(apple_image_path)

    print_task_title('4. 이미지 이진화, 에지 검출, 블러링')
    task_4_binary_edge_blur(apple_image_path, orange_image_path)
    task_4_bonus_partial_blur(orange_image_path)

    print_task_title('5. HSV 변환 및 채널 출력')
    task_5_hsv_channels(apple_image_path)
    task_5_bonus_bgr_channels(apple_image_path)

    print_task_title('6. 객체 표시와 라벨링')
    task_6_object_labeling(orange_image_path)
    task_6_bonus_shape_labeling(orange_image_path)

    print()
    print('모든 작업이 종료되었습니다.')


if __name__ == '__main__':
    main()