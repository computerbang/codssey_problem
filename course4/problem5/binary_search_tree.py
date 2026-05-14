'''Binary Search Tree implementation using only Python built-ins.

PEP 8 스타일 가이드를 준수하며, 외부 라이브러리를 사용하지 않습니다.
'''


class Node:
    '''이진 탐색 트리의 노드를 표현하는 클래스.'''

    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinarySearchTree:
    '''insert, find, delete 기능을 제공하는 이진 탐색 트리 클래스.'''

    def __init__(self):
        self.root = None

    def insert(self, value):
        '''트리에 새로운 값을 삽입한다. 중복 값은 무시된다.'''
        self.root = self._insert_node(self.root, value)

    def _insert_node(self, node, value):
        if node is None:
            return Node(value)
        if value < node.value:
            node.left = self._insert_node(node.left, value) # 재귀함수(recursive function)
        elif value > node.value:
            node.right = self._insert_node(node.right, value)
        return node

    def find(self, value):
        '''값이 트리에 존재하면 True, 그렇지 않으면 False를 반환한다.'''
        return self._find_node(self.root, value) is not None

    def _find_node(self, node, value):
        if node is None:
            return None
        if value == node.value:
            return node
        if value < node.value:
            return self._find_node(node.left, value)
        return self._find_node(node.right, value)

    def delete(self, value):
        '''트리에서 특정 값을 삭제한다. 존재하지 않으면 아무 동작도 하지 않는다.'''
        self.root = self._delete_node(self.root, value)

    def _delete_node(self, node, value):
        if node is None:
            return None
        if value < node.value:
            node.left = self._delete_node(node.left, value)
        elif value > node.value:
            node.right = self._delete_node(node.right, value)
        else:
            # 삭제할 노드를 찾은 경우
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            # 자식이 둘인 경우 오른쪽 서브트리의 최솟값(중위 후속자)으로 대체
            successor = self._find_min(node.right)
            node.value = successor.value
            node.right = self._delete_node(node.right, successor.value)
        return node

    def _find_min(self, node):
        current = node
        while current.left is not None:
            current = current.left
        return current

    def inorder(self):
        '''중위 순회 결과를 리스트로 반환한다 (정렬된 순서).'''
        result = []
        self._inorder_traverse(self.root, result)
        return result

    def _inorder_traverse(self, node, result):
        if node is None:
            return
        self._inorder_traverse(node.left, result)
        result.append(node.value)
        self._inorder_traverse(node.right, result)


if __name__ == '__main__':
    # 과제 요구사항에 따라 'binarytree'라는 이름으로 인스턴스 생성
    binarytree = BinarySearchTree()

    # 삽입 테스트
    values = [50, 30, 70, 20, 40, 60, 80, 35]
    for v in values:
        binarytree.insert(v)
    print('중위 순회 결과:', binarytree.inorder())

    # 탐색 테스트
    print('40 존재 여부:', binarytree.find(40))
    print('100 존재 여부:', binarytree.find(100))

    # 삭제 테스트 (자식 둘인 노드)
    binarytree.delete(30)
    print('30 삭제 후:', binarytree.inorder())

    # 삭제 테스트 (루트 노드)
    binarytree.delete(50)
    print('50(루트) 삭제 후:', binarytree.inorder())

    # 삭제 테스트 (리프 노드)
    binarytree.delete(80)
    print('80(리프) 삭제 후:', binarytree.inorder())

    # 존재하지 않는 값 삭제 (오류 없이 처리)
    binarytree.delete(999)
    print('999(없는 값) 삭제 후:', binarytree.inorder())