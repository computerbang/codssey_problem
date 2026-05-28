MAX_SIZE = 10


class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        if len(self.items) >= MAX_SIZE:
            print('경고: 스택이 가득 찼습니다. 더 이상 추가할 수 없습니다.')
            return

        self.items.append(item)
        print(f'push 완료: {item}')

    def pop(self):
        if self.empty():
            print('경고: 스택이 비어 있습니다. 가져올 내용이 없습니다.')
            return None

        item = self.items.pop()
        print(f'pop 완료: {item}')

        return item

    def empty(self):
        return len(self.items) == 0

    def peek(self):
        if self.empty():
            print('경고: 스택이 비어 있습니다. 확인할 내용이 없습니다.')
            return None

        item = self.items[-1]
        print(f'peek 결과: {item}')

        return item

    def show_stack(self):
        print()
        print('===== 현재 스택 상태 =====')

        if self.empty():
            print('|       empty       |')
            print('======================')
            return

        for index in range(len(self.items) - 1, -1, -1):
            item = self.items[index]

            if index == len(self.items) - 1:
                print(f'| {item:<16} | <- top')
            else:
                print(f'| {item:<16} |')

        print('======================')
        print(f'현재 데이터 개수: {len(self.items)} / {MAX_SIZE}')
        print()


def test_stack():
    stack = Stack()

    print('===== 스택 데이터 추가 테스트 =====')

    for number in range(1, 12):
        stack.push(f'data-{number}')
        stack.show_stack()

    print('===== peek 테스트 =====')
    stack.peek()
    stack.show_stack()

    print('===== pop 테스트 =====')

    for _ in range(1, 12):
        stack.pop()
        stack.show_stack()

    print('===== empty 테스트 =====')

    if stack.empty():
        print('empty 결과: 스택이 비어 있습니다.')
    else:
        print('empty 결과: 스택에 데이터가 있습니다.')


def main():
    test_stack()


if __name__ == '__main__':
    main()