import heapq

class Node:
    def __init__(self, x, y, cost, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)

def a_star_search(start, goal, grid):
    # Create the start and goal nodes
    start_node = Node(start[0], start[1], grid[start[0]][start[1]])
    goal_node = Node(goal[0], goal[1], grid[goal[0]][goal[1]])

    open_list = []
    closed_list = set()

    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.x == goal_node.x and current_node.y == goal_node.y:
            path = []
            while current_node is not None:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_list.add((current_node.x, current_node.y))

        for x, y in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor_x = current_node.x + x
            neighbor_y = current_node.y + y

            if neighbor_x < 0 or neighbor_x >= len(grid) or neighbor_y < 0 or neighbor_y >= len(grid[0]):
                continue

            if grid[neighbor_x][neighbor_y] == -1 or (neighbor_x, neighbor_y) in closed_list:
                continue

            neighbor_node = Node(neighbor_x, neighbor_y, grid[neighbor_x][neighbor_y], current_node)

            neighbor_node.g = current_node.g + neighbor_node.cost

            neighbor_node.h = heuristic(neighbor_node, goal_node)

            neighbor_node.f = neighbor_node.g + neighbor_node.h

            heapq.heappush(open_list, neighbor_node)

    return None

start = (0, 0)
goal = (2,2)
grid = [[0, 1, 2, 3],
        [1, 2, -1, 4],
        [2, -1, 4, 5],
        [3, 4, 5, 6]]

path = a_star_search(start, goal, grid)
print(path)
