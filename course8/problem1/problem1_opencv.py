import cv2


def show_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print('이미지를 불러올 수 없습니다.')
        return

    cv2.imshow('Image Viewer', image)

    print('\n===== 이미지 출력 =====')
    print('아무 키나 누르면 이미지 창이 닫힙니다.')
    print('=======================\n')

    cv2.waitKey(0)
    cv2.destroyWindow('Image Viewer')


def show_video(video_path):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print('동영상을 열 수 없습니다.')
        return

    print('\n===== 동영상 출력 =====')
    print('ESC 키를 누르면 동영상 재생이 종료됩니다.')
    print('=======================\n')

    while True:
        success, frame = capture.read()

        if not success:
            print('영상 재생 종료')
            break

        cv2.imshow('Video Player', frame)

        key = cv2.waitKey(33) & 0xFF

        if key == 27:
            print('동영상 출력 종료')
            break

    capture.release()
    cv2.destroyWindow('Video Player')


def show_macbook_camera():
    capture = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

    if not capture.isOpened():
        print('카메라를 열 수 없습니다.')
        return

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print('\n===== 맥북 카메라 실시간 출력 =====')
    print('ESC 키를 누르면 카메라 출력이 종료됩니다.')
    print('==================================\n')

    while True:
        success, frame = capture.read()

        if not success:
            print('카메라 프레임을 읽을 수 없습니다.')
            break

        cv2.imshow('MacBook Camera Viewer', frame)

        key = cv2.waitKey(33) & 0xFF

        if key == 27:
            print('카메라 출력 종료')
            break

    capture.release()
    cv2.destroyWindow('MacBook Camera Viewer')


def main():
    image_path = 'apple.jpg'
    video_path = 'video.mp4'

    print('===== 문제 1번 OpenCV 기본 과제 시작 =====')

    show_image(image_path)
    show_video(video_path)
    show_macbook_camera()

    cv2.destroyAllWindows()

    print('===== 문제 1번 프로그램 종료 =====')


if __name__ == '__main__':
    main()