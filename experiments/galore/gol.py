import time
import os
import random
import matplotlib.pyplot as plt

class GameOfLife:
    def __init__(self, grid_size=6, live_probability=0.2):
        self.grid_size = grid_size
        self.live_probability = live_probability
        self.grid = self.random_grid()

    def random_grid(self):
        return [[1 if random.random() < self.live_probability else 0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

    def print_grid(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(self.get_board())

    def next_generation(self):
        new_grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                live_neighbors = self.count_live_neighbors(i, j)
                if self.grid[i][j] == 1 and live_neighbors in (2, 3):
                    new_grid[i][j] = 1
                elif self.grid[i][j] == 0 and live_neighbors == 3:
                    new_grid[i][j] = 1
        self.grid = new_grid

    def count_live_neighbors(self, x, y):
        count = 0
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dx, dy in directions:
            nx, ny = (x + dx) % self.grid_size, (y + dy) % self.grid_size
            count += self.grid[nx][ny]
        return count

    def reinit(self, p=None):
        self.live_probability = p or self.live_probability
        self.grid = self.random_grid()

    def get_board(self):
        return '\n'.join(' '.join('*' if cell else '.' for cell in row) for row in self.grid)

    def run(self):
        while True:
            self.print_grid()
            self.next_generation()
            time.sleep(0.1)

if __name__ == '__main__':
    grid_size = 10
    p = 0.4  # Set the initial probability for cells being alive
    game = GameOfLife(grid_size=grid_size, live_probability=p)
    game.reinit(p=0.4)
    plst = []
    NUM_GAMES = 1000
    for p_idx in range(1,20):
        p = 0.05 * p_idx
        tot = 0
        for _ in range(NUM_GAMES):
            lst = []
            game.reinit(p)
            for _ in range(100):
                # os.system('cls' if os.name == 'nt' else 'clear')
                state = game.get_board()
                # print(state)
                
                game.next_generation()
                if state in lst:
                    lst.append(state)
                    break
                lst.append(state)
                # time.sleep(0.1)
            # print(lst)
            tot += len(lst)
        print(p_idx,tot/NUM_GAMES)
        if p_idx%10 == 0:
            time.sleep(1)
        plst.append(tot/NUM_GAMES)
    print(plst)
    plt.plot(plst)
    plt.show()