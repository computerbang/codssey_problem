# pip install nava

from nava import play, stop
import time

## 글로벌 변수 선언
head = None
posHEAD = 0  # 리스트 처음 위치
posTAIL = 1  # 리스트 마지막 위치
poseNODE = 2  # 리스트 중간 위치


class Node:
    def __init__(self, data, filepath=None):
        self.data = data          # 곡 이름
        self.filepath = filepath  # 음악 파일 경로 (.wav)
        self.next = None


class linkedlist:
    def __init__(self):
        self.head = None

    def insert(self, data, position=posHEAD, node=None, filepath=None):
        global head
        new_node = Node(data, filepath)

        # 1. 리스트가 비어있는 경우
        if head is None:
            head = new_node
            return

        # 2. 리스트의 처음에 삽입하는 경우
        if position == posHEAD:
            new_node.next = head
            head = new_node
            return

        # 3. 리스트의 마지막에 삽입하는 경우
        if position == posTAIL:
            current = head
            while current.next is not None:
                current = current.next
            current.next = new_node
            return

        # 4. 리스트의 중간에 삽입하는 경우
        if position == poseNODE:
            if node is None:
                raise ValueError("중간 위치에 삽입하려면 노드를 지정해야 합니다.")
            current = head
            while current is not None and current.data != node.data:
                current = current.next
            if current is None:
                raise ValueError("지정된 노드를 찾을 수 없습니다.")
            new_node.next = current.next
            current.next = new_node

    def delete(self, position=posHEAD, node=None):
        global head

        # 1. 리스트가 비어있는 경우
        if head is None:
            raise ValueError("리스트가 비어있습니다.")

        # 2. 리스트의 처음을 삭제하는 경우
        if position == posHEAD:
            head = head.next
            return

        # 3. 리스트의 마지막을 삭제하는 경우
        if position == posTAIL:
            current = head
            prev = None
            while current.next is not None:
                prev = current
                current = current.next
            if prev is not None:
                prev.next = None
            else:
                head = None
            return

        # 4. 리스트의 중간을 삭제하는 경우
        if position == poseNODE:
            if node is None:
                raise ValueError("중간 위치에서 삭제하려면 노드를 지정해야 합니다.")
            current = head
            prev = None
            while current is not None and current.data != node.data:
                prev = current
                current = current.next
            if current is None:
                raise ValueError("지정된 노드를 찾을 수 없습니다.")
            if prev is not None:
                prev.next = current.next
            else:
                head = current.next

    def get_list(self):
        """처음부터 끝까지 순차적으로 가져오는 함수 (보너스 과제)"""
        global head
        result = []
        current = head
        idx = 1
        while current:
            result.append(f"{idx}. {current.data}")
            current = current.next
            idx += 1
        return result

    def display(self):
        global head
        current = head
        if current is None:
            print("  리스트가 비어있습니다.")
            return
        idx = 1
        while current:
            marker = "♪" if current.filepath else "♩"
            print(f"  {marker} {idx}. {current.data}")
            current = current.next
            idx += 1

    def play_song(self, song_name):
        """nava를 사용하여 특정 곡 재생"""
        global head
        current = head
        while current is not None:
            if current.data == song_name:
                if current.filepath:
                    print(f"  ▶ 재생 중: {current.data}")
                    try:
                        play(current.filepath)
                    except Exception as e:
                        print(f"  ⚠ 재생 오류: {e}")
                else:
                    print(f"  ⚠ '{current.data}'에 음악 파일이 지정되지 않았습니다.")
                return
            current = current.next
        print(f"  ⚠ '{song_name}'을(를) 찾을 수 없습니다.")

    def play_all(self, duration=3):
        """전체 플레이리스트 순차 재생 (각 곡 duration초)"""
        global head
        current = head
        if current is None:
            print("  리스트가 비어있습니다.")
            return
        while current:
            if current.filepath:
                print(f"  ▶ 재생 중: {current.data}")
                try:
                    sound_id = play(current.filepath, async_mode=True)
                    time.sleep(duration)
                    stop(sound_id)
                except Exception as e:
                    print(f"  ⚠ 재생 오류: {e}")
            else:
                print(f"  ⏭ 건너뜀 (파일 없음): {current.data}")
            current = current.next
        print("  ■ 재생 완료")

    def load_list(self, filename):
        try:
            with open(filename, 'r') as file:
                for line in file:
                    self.insert(line.strip(), position=posTAIL)
            print(f"  {filename}에서 데이터를 성공적으로 불러왔습니다.")
        except FileNotFoundError:
            print(f"  {filename} 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"  데이터를 불러오는 중 오류가 발생했습니다: {e}")


# ====================================================
# 테스트: 음악 플레이리스트
# ====================================================
if __name__ == "__main__":
    pl = linkedlist()

    print("=" * 50)
    print("  음악 플레이리스트 - 단순 연결 리스트")
    print("=" * 50)

    # --- 곡 추가 (맨 끝에) ---
    print("\n[1] 곡 3개 추가 (TAIL)")
    pl.insert("Bohemian Rhapsody - Queen", posTAIL)
    pl.insert("Hotel California - Eagles", posTAIL)
    pl.insert("Imagine - John Lennon", posTAIL)
    pl.display()

    # --- 곡 추가 (맨 앞에) ---
    print("\n[2] 맨 앞에 추가 (HEAD)")
    pl.insert("Billie Jean - Michael Jackson", posHEAD)
    pl.display()

    # --- 곡 추가 (중간에) ---
    print("\n[3] 'Bohemian Rhapsody' 뒤에 삽입 (NODE)")
    pl.insert("Yesterday - The Beatles", poseNODE,
              node=Node("Bohemian Rhapsody - Queen"))
    pl.display()

    # --- 곡 삭제 (중간) ---
    print("\n[4] 'Hotel California' 삭제 (NODE)")
    pl.delete(poseNODE, node=Node("Hotel California - Eagles"))
    pl.display()

    # --- 곡 삭제 (맨 앞) ---
    print("\n[5] 맨 앞 곡 삭제 (HEAD)")
    pl.delete(posHEAD)
    pl.display()

    # --- 곡 삭제 (맨 끝) ---
    print("\n[6] 맨 끝 곡 삭제 (TAIL)")
    pl.delete(posTAIL)
    pl.display()

    # --- 보너스: get_list() ---
    print("\n[7] get_list() 결과:")
    for item in pl.get_list():
        print(f"  {item}")

    # --- nava 재생 예시 (wav 파일이 있을 때) ---
    print("\n[8] nava 재생 테스트")
    head = None  # 리스트 초기화
    pl2 = linkedlist()
    pl2.insert("sample_song", posTAIL, filepath="sample.wav")
    pl2.play_song("sample_song")
    # 파일이 없으면 오류 메시지 출력