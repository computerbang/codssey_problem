class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data, index=None):
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
            return

        if index == 0:
            new_node.next = self.head
            self.head = new_node
            return

        current = self.head
        position = 0

        if index is None:
            while current.next is not None:
                current = current.next
            current.next = new_node
            return

        while current.next is not None and position < index - 1:
            current = current.next
            position += 1

        new_node.next = current.next
        current.next = new_node

    def delete(self, data):
        if self.head is None:
            return False

        if self.head.data == data:
            self.head = self.head.next
            return True

        current = self.head

        while current.next is not None:
            if current.next.data == data:
                current.next = current.next.next
                return True
            current = current.next

        return False

    def get_list(self):
        result = []
        current = self.head

        while current is not None:
            result.append(current.data)
            current = current.next

        return result


class CircularList:
    def __init__(self):
        self.head = None
        self.current = None

    def insert(self, data, index=None):
        new_node = Node(data)

        if self.head is None:
            self.head = new_node
            new_node.next = self.head
            self.current = self.head
            return

        if index == 0:
            tail = self.head

            while tail.next != self.head:
                tail = tail.next

            new_node.next = self.head
            tail.next = new_node
            self.head = new_node
            return

        current = self.head
        position = 0

        if index is None:
            while current.next != self.head:
                current = current.next

            current.next = new_node
            new_node.next = self.head
            return

        while current.next != self.head and position < index - 1:
            current = current.next
            position += 1

        new_node.next = current.next
        current.next = new_node

    def delete(self, data):
        if self.head is None:
            return False

        if self.head.data == data:
            if self.head.next == self.head:
                self.head = None
                self.current = None
                return True

            tail = self.head

            while tail.next != self.head:
                tail = tail.next

            tail.next = self.head.next
            self.head = self.head.next
            self.current = self.head
            return True

        current = self.head

        while current.next != self.head:
            if current.next.data == data:
                current.next = current.next.next
                return True
            current = current.next

        return False

    def get_next(self):
        if self.current is None:
            return None

        data = self.current.data
        self.current = self.current.next

        return data

    def search(self, data):
        if self.head is None:
            return -1

        current = self.head
        index = 0

        while True:
            if current.data == data:
                return index

            current = current.next
            index += 1

            if current == self.head:
                break

        return -1

    def get_list(self):
        result = []

        if self.head is None:
            return result

        current = self.head

        while True:
            result.append(current.data)
            current = current.next

            if current == self.head:
                break

        return result


def main():
    linkedlist = LinkedList()

    linkedlist.insert('Ditto')
    linkedlist.insert('Hype Boy')
    linkedlist.insert('ETA')
    linkedlist.insert('Super Shy', 0)
    linkedlist.insert('OMG', 2)

    print('단순 연결 리스트 목록')
    print(linkedlist.get_list())

    linkedlist.delete('ETA')

    print('ETA 삭제 후')
    print(linkedlist.get_list())

    circularlist = CircularList()

    circularlist.insert('Seven')
    circularlist.insert('Love Lee')
    circularlist.insert('Drama')
    circularlist.insert('I AM')
    circularlist.insert('Attention', 2)

    print('원형 연결 리스트 목록')
    print(circularlist.get_list())

    print('순차 재생')
    print(circularlist.get_next())
    print(circularlist.get_next())
    print(circularlist.get_next())
    print(circularlist.get_next())
    print(circularlist.get_next())
    print(circularlist.get_next())

    search_title = 'Drama'
    search_result = circularlist.search(search_title)

    if search_result == -1:
        print(search_title + '을 찾을 수 없습니다.')
    else:
        print(search_title + '의 위치는 ' + str(search_result) + '번입니다.')

    circularlist.delete('Love Lee')

    print('Love Lee 삭제 후')
    print(circularlist.get_list())


if __name__ == '__main__':
    main()