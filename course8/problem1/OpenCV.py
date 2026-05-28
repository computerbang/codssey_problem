import cv2
from datetime import datetime


WAIT_TIME = 33

ESC_KEY = 27
CTRL_Z_KEY = 26
CTRL_X_KEY = 24
C_KEY = ord('c')


class VideoController:
    def __init__(self, video_path):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(self.video_path)

        self.writer = None
        self.is_recording = False

        if not self.capture.isOpened():
            print('동영상을 열 수 없습니다.')
            self.is_opened = False
            return

        self.is_opened = True

        self.width = int(
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        self.fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )

        if self.fps <= 0:
            self.fps = 30.0

    def create_file_name(self, extension):
        current_time = datetime.now()

        return current_time.strftime(
            f'%Y_%m_%d_%H-%M-%S.{extension}'
        )

    def capture_image(self, frame):
        image_name = self.create_file_name('png')

        cv2.imwrite(image_name, frame)

        print(f'이미지 저장 완료: {image_name}')

    def start_recording(self):
        if self.is_recording:
            print('이미 녹화 중입니다.')
            return

        video_name = self.create_file_name('avi')

        codec = cv2.VideoWriter_fourcc(*'XVID')

        self.writer = cv2.VideoWriter(
            video_name,
            codec,
            self.fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            print('동영상 녹화 파일을 생성할 수 없습니다.')
            self.writer = None
            return

        self.is_recording = True

        print(f'녹화 시작: {video_name}')
        print('사용 코덱: XVID')

    def stop_recording(self):
        if not self.is_recording:
            print('현재 녹화 중이 아닙니다.')
            return

        self.is_recording = False

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        print('녹화 종료')

    def play_video(self):
        if not self.is_opened:
            return

        print('\n===== 단축키 안내 =====')
        print('ESC      : 프로그램 종료')
        print('Ctrl+Z   : 이미지 캡처')
        print('Ctrl+X   : 녹화 시작')
        print('C        : 녹화 종료')
        print('======================\n')

        while True:
            success, frame = self.capture.read()

            if not success:
                print('영상 재생 종료')
                break

            cv2.imshow('Video Player', frame)

            if self.is_recording and self.writer is not None:
                self.writer.write(frame)

            key = cv2.waitKey(WAIT_TIME) & 0xFF

            if key == ESC_KEY:
                print('프로그램 종료')
                break

            elif key == CTRL_Z_KEY:
                self.capture_image(frame)

            elif key == CTRL_X_KEY:
                self.start_recording()

            elif key == C_KEY:
                self.stop_recording()

        self.release()

    def release(self):
        if self.is_recording:
            self.stop_recording()

        if self.capture is not None:
            self.capture.release()

        if self.writer is not None:
            self.writer.release()
            self.writer = None

        cv2.destroyAllWindows()


def show_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print('이미지를 불러올 수 없습니다.')
        return

    cv2.imshow('Image Viewer', image)

    print('\n===== 이미지 출력 =====')
    print('아무 키나 누르면 창이 닫힙니다.')
    print('=======================\n')

    cv2.waitKey(0)

    cv2.destroyAllWindows()


def convert_video_with_codec(video_path, codec_name, extension):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print('동영상을 열 수 없습니다.')
        return

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    current_time = datetime.now()
    output_name = current_time.strftime(
        f'%Y_%m_%d_%H-%M-%S_{codec_name}.{extension}'
    )

    codec = cv2.VideoWriter_fourcc(*codec_name)

    writer = cv2.VideoWriter(
        output_name,
        codec,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print(f'{codec_name} 코덱으로 파일을 생성할 수 없습니다.')
        capture.release()
        return

    while True:
        success, frame = capture.read()

        if not success:
            break

        writer.write(frame)

    writer.release()
    capture.release()

    print(f'{codec_name} 코덱 저장 완료: {output_name}')


def test_two_codecs(video_path):
    convert_video_with_codec(video_path, 'XVID', 'avi')
    convert_video_with_codec(video_path, 'mp4v', 'mp4')


def main():
    image_path = 'apple.jpg'
    video_path = 'video.mp4'

    print('===== OpenCV 과제 시작 =====')

    show_image(image_path)

    controller = VideoController(video_path)
    controller.play_video()

    test_two_codecs(video_path)

    print('===== 프로그램 종료 =====')


if __name__ == '__main__':
    main()