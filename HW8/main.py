import numpy as np
from pprint import pprint
from typing import List, Union

path_to_goal = []
sample_move_grid = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 8, 1, 0],
    [4, 6, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0],
    [1, 6, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 6, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 0],
    [1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 0],
    [1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 0],
    [4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 0],
    [1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 0],
    [1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 0],
    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
]
visited = np.zeros_like(sample_move_grid)

# This is a stack that will store the path found.
dfs_path = []
                
def search_move_grid_dfs(
        move_grid:List[List[int]], 
        cur_r:int, 
        cur_c:int, 
        target:int
    )-> Union[bool, None]: 
    """Depth first search algorithm to solve the maze puzzle.

    Args:
        move_grid (List[List[int]]): the matrix maze representation
        cur_r (int): index of the current row
        cur_c (int): index of the current column
        target (int): integer target to find

    Returns:
        (bool or None): whether or not the target has been reached
    """
    # Update the path stack.
    dfs_path.append((cur_r, cur_c))

    # Return if indices are out of bounds or if cell has been visited.
    if cur_r < 0 or cur_c < 0 or cur_r > 11 or cur_c > 19:
        return
    if visited[cur_r][cur_c] == 1:
        return
    
    visited[cur_r][cur_c] = 1

    # If at the target, return True.
    if move_grid[cur_r][cur_c] == target:
        return True

    # If 0, search left.
    elif move_grid[cur_r][cur_c] == 0:
        found = search_move_grid_dfs(sample_move_grid, cur_r, cur_c-1, target)
        if found: return found
        else: dfs_path.pop()

        return False

    # If 1, possibly search left, right, and down.
    elif move_grid[cur_r][cur_c] == 1:
        val = sample_move_grid[cur_r+1][cur_c-1]
        if val == 4 or val == 6:
            found = search_move_grid_dfs(sample_move_grid, cur_r, cur_c-1, target)
            if found: return found
            else: dfs_path.pop()

        val = sample_move_grid[cur_r+1][cur_c+1]
        if val == 4 or val == 6:
            found = search_move_grid_dfs(sample_move_grid, cur_r, cur_c+1, target)
            if found: return found
            else: dfs_path.pop()

        val = sample_move_grid[cur_r+1][cur_c]
        if val == 6:
            found = search_move_grid_dfs(sample_move_grid, cur_r+1, cur_c, target)
            if found: return found
            else: dfs_path.pop()

        return False

    # If 6, search up and down, possibly left and right.
    elif move_grid[cur_r][cur_c] == 6:
        found = search_move_grid_dfs(sample_move_grid, cur_r+1, cur_c, target)
        if found: return found
        else: dfs_path.pop()

        found = search_move_grid_dfs(sample_move_grid, cur_r-1, cur_c, target)
        if found: return found
        else: dfs_path.pop()

        val = sample_move_grid[cur_r+1][cur_c]
        if val == 4:
            found = search_move_grid_dfs(sample_move_grid, cur_r, cur_c-1, target)
            if found: return found
            else: dfs_path.pop()

            found = search_move_grid_dfs(sample_move_grid, cur_r, cur_c+1, target)
            if found: return found
            else: dfs_path.pop()

        return False

def search_move_grid_bfs(
        move_grid:List[List[int]], 
        cur_r:int, 
        cur_c:int, 
        target:int
    )-> List[int]: 
    """Breadth first search algorithm to solve the maze puzzle.

    Args:
        move_grid (List[List[int]]): the matrix maze representation
        cur_r (int): index of the current row
        cur_c (int): index of the current column
        target (int): integer target to find

    Returns:
        (List[int]): the path taken by the BFS algorithm
    """

    def process_path_dict(r, c):
        """Build the path taken by the algorithm by populating a list.
        """
        if r is None and c is None:
            return
        bfs_path_list.append((r, c))
        process_path_dict(bfs_path[(r, c)][0], bfs_path[(r, c)][1])

    # Return if indices are out of bounds.
    if cur_r < 0 or cur_c < 0 or cur_r > 11 or cur_c > 19:
        return

    # Record of path taken that will eventually be returned.
    bfs_path_list = []

    # Used as a pointer system to track parents.
    bfs_path = dict()

    # Contains the next cell to explore and the cell's parent.
    queue = [(cur_r, cur_c, None, None)]

    while queue:
        # Get this element's row and col and this element's parent.
        cur_r, cur_c = queue[0][0], queue[0][1] 
        par_r, par_c = queue[0][2], queue[0][3]

        # Remove this element from the list.
        queue = queue[1:]

        # If visited already, skip, else mark as visited.
        if visited[cur_r][cur_c] == 1:
            continue
        visited[cur_r][cur_c] = 1
        bfs_path[(cur_r, cur_c)] = (par_r, par_c)

        # If at the target, stop searching and return the path found.
        if move_grid[cur_r][cur_c] == target:
            process_path_dict(cur_r, cur_c)
            return bfs_path_list

        # If 0, search left.
        elif move_grid[cur_r][cur_c] == 0:
            queue.append((cur_r, cur_c-1, cur_r, cur_c))

        # If 1, possibly search left, right, and down.
        elif move_grid[cur_r][cur_c] == 1:
            val = sample_move_grid[cur_r+1][cur_c-1]
            if val == 4 or val == 6:
                queue.append((cur_r, cur_c-1, cur_r, cur_c))

            val = sample_move_grid[cur_r+1][cur_c+1]
            if val == 4 or val == 6:
                queue.append((cur_r, cur_c+1, cur_r, cur_c))

            val = sample_move_grid[cur_r+1][cur_c]
            if val == 6:
                queue.append((cur_r+1, cur_c, cur_r, cur_c))

        # If 6, search up and down, possibly left and right.
        elif move_grid[cur_r][cur_c] == 6:
            queue.append((cur_r+1, cur_c, cur_r, cur_c))
            queue.append((cur_r-1, cur_c, cur_r, cur_c))

            val = sample_move_grid[cur_r+1][cur_c]
            if val == 4:
                queue.append((cur_r, cur_c-1, cur_r, cur_c))
                queue.append((cur_r, cur_c+1, cur_r, cur_c))

def test_dfs_sol():
    global visited
    global dfs_path

    visited, dfs_path = np.zeros_like(sample_move_grid), []
    init_row, init_col, target = 10, 19, 8
    print(f"\n\nSearching for {target} from: {init_row, init_col}")
    search_move_grid_dfs(sample_move_grid, init_row, init_col, target)
    pprint(dfs_path)
    print("-----------------------------------------------------------")

    visited, dfs_path = np.zeros_like(sample_move_grid), []
    init_row, init_col, target = 10, 0, 8
    print(f"\n\nSearching for {target} from: {init_row, init_col}")
    search_move_grid_dfs(sample_move_grid, init_row, init_col, target)
    pprint(dfs_path)
    print("-----------------------------------------------------------")

    visited, dfs_path = np.zeros_like(sample_move_grid), []
    init_row, init_col, target = 1, 0, 8
    print(f"\n\nSearching for {target} from: {init_row, init_col}")
    search_move_grid_dfs(sample_move_grid, init_row, init_col, target)
    pprint(dfs_path)
    print("-----------------------------------------------------------")

def test_bfs_sol():
    global visited
    
    visited = np.zeros_like(sample_move_grid)
    init_row, init_col, target = 10, 19, 8
    print(f"\n\nSearching for {target} from: {init_row, init_col}")
    bfs_path = search_move_grid_bfs(sample_move_grid, init_row, init_col, target)
    pprint(bfs_path)
    print("-----------------------------------------------------------")
    
    visited = np.zeros_like(sample_move_grid)
    init_row, init_col, target = 10, 0, 8
    print(f"\n\nSearching for {target} from: {init_row, init_col}")
    bfs_path = search_move_grid_bfs(sample_move_grid, init_row, init_col, target)
    pprint(bfs_path)
    print("-----------------------------------------------------------")

    visited = np.zeros_like(sample_move_grid)
    init_row, init_col, target = 1, 0, 8
    print(f"\n\nSearching for {target} from: {init_row, init_col}")
    bfs_path = search_move_grid_bfs(sample_move_grid, init_row, init_col, target)
    pprint(bfs_path)
    print("-----------------------------------------------------------")

def main():
    test_bfs_sol()
    test_dfs_sol()

if __name__ == "__main__":
    main()
    


