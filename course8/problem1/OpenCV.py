import cv2
from datetime import datetime


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

    def create_file_name(self, extension):
        current_time = datetime.now()

        return current_time.strftime(
            f'%Y-%m-%d_%H-%M-%S.{extension}'
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

        # 코덱 1 : XVID
        codec = cv2.VideoWriter_fourcc(*'XVID')

        self.writer = cv2.VideoWriter(
            video_name,
            codec,
            self.fps,
            (self.width, self.height)
        )

        self.is_recording = True

        print(f'녹화 시작: {video_name}')

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
        print('C   : 녹화 종료')
        print('======================\n')

        while True:
            success, frame = self.capture.read()

            if not success:
                print('영상 재생 종료')
                break

            cv2.imshow('Video Player', frame)

            if self.is_recording and self.writer is not None:
                self.writer.write(frame)

            key = cv2.waitKey(33) & 0xFF

            # ESC
            if key == 27:
                print('프로그램 종료')
                break

            # Ctrl + Z
            elif key == 26:
                self.capture_image(frame)

            # Ctrl + X
            elif key == 24:
                self.start_recording()

            # C 키
            elif key == ord('c'):
                self.stop_recording()
                
    def release(self):
        self.capture.release()

        if self.writer is not None:
            self.writer.release()

        cv2.destroyAllWindows()


class CameraViewer:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)

        if not self.capture.isOpened():
            print('카메라를 열 수 없습니다.')
            self.is_opened = False
            return

        self.is_opened = True

        # 해상도 설정
        self.capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            640
        )

        self.capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            480
        )

    def show_camera(self):
        if not self.is_opened:
            return

        print('\n===== 카메라 실행 =====')
        print('ESC 키를 누르면 종료됩니다.')
        print('=======================\n')

        while True:
            success, frame = self.capture.read()

            if not success:
                print('카메라 프레임을 읽을 수 없습니다.')
                break

            cv2.imshow('Camera Viewer', frame)

            key = cv2.waitKey(33) & 0xFF

            if key == 27:
                print('카메라 종료')
                break

        self.capture.release()

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


def main():
    image_path = 'apple.jpg'
    video_path = 'video.mp4'

    print('===== OpenCV 과제 시작 =====')

    # 이미지 출력
    show_image(image_path)

    # 영상 재생 및 제어
    controller = VideoController(video_path)
    controller.play_video()

    # 카메라 출력
    camera = CameraViewer()
    camera.show_camera()

    print('===== 프로그램 종료 =====')


if __name__ == '__main__':
    main()
