from typing import Callable
from typing import Callable, List

import copy

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


class HookaJeeves:
    def __init__(self, object_func: Callable, start_point: np.array, start_step_len: np.array, eps: float, lambd: float, alph: float):
        self.object_func = object_func
        self.start_point = np.copy(start_point)
        self.start_step_len = np.copy(start_step_len)
        self.eps = eps
        self.lambd = lambd
        self.alph = alph

    def explore_search(self, base_point: List[float], step_len: List[float]) -> List[float]:
        base_val = self.object_func(*base_point)
        
        res_point = np.copy(base_point)
        for coord in range(len(base_point)):            
            res_point[coord] += step_len[coord]

            res_val = self.object_func(*res_point)
            if res_val >= base_val:
                res_point[coord] -= 2 * step_len[coord]
                
                res_val = self.object_func(*res_point)
                if res_val >= base_val:
                    res_point[coord] += step_len[coord]
        
        return res_point

    def pattern_search_step(self, prev_point: np.array, cur_point: np.array) -> np.array:
        cand_point = copy.copy(cur_point)
        for coord in range(len(cur_point)):
            cand_point[coord] += self.lambd * (cur_point[coord] - prev_point[coord])

        return cand_point
    
    def can_decrease_step(self, step_len: np.array) -> bool:
        for coord in range(len(step_len)):
            if step_len[coord] >= self.eps:
                return True
            
        return False

    def decrease_step(self, step_len: np.array) -> None:
        for coord in range(len(step_len)):
            if step_len[coord] >= self.eps:
                step_len[coord] /= self.alph
        

    def solve(self) -> List[np.array]:
        step_len = self.start_step_len
        result_points = [self.start_point]
        base_point = self.start_point

        while True:
            new_base_point = self.explore_search(base_point, step_len)
            new_base_val = self.object_func(*new_base_point)
            base_val = self.object_func(*base_point)
            if new_base_val >= base_val:
                can_decrease = self.can_decrease_step(step_len)
                if not can_decrease:
                    return result_points
                self.decrease_step(step_len)
                continue

            result_points.append(new_base_point)
            
            search_point = self.pattern_search_step(base_point, new_base_point)
            explored_after_search_point = self.explore_search(search_point, step_len)

            explored_after_search_val = self.object_func(*explored_after_search_point)
            if explored_after_search_val < new_base_val:
                base_point = explored_after_search_point 
            else:
                base_point = new_base_point


            
        


# Определяем функцию f(x1, x2)
def f(x1: np.array, x2: np.array):
    return np.square(x1 + 1) + np.square(x2)


# Создаем сетку значений x1 и x2
x1 = np.linspace(-10, 10, 10)
x2 = np.linspace(-10, 10, 10)
X_tuple = np.meshgrid(x1, x2)

X = np.array(X_tuple).reshape(2, -1).T

# Вычисляем значения функции на сетке
FX = f(X[:, 0], X[:, 1])

fig = plt.figure(figsize=(8, 6))

# Строим график функции
ax = fig.add_subplot(2, 3, 2, projection='3d')
ax.plot_trisurf(X[:, 0], X[:, 1], FX)
# Настройки графика
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('График функции f(x1, x2)')


ax = fig.add_subplot(2, 3, 4)

hj = HookaJeeves(f, [10, 7], [1, 1], 0.01, 1, 2)

search_res = hj.solve()
x1_res = [x[0] for x in search_res]
x2_res = [x[1] for x in search_res]

ax.plot(x1_res, x2_res, "bo--")
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_xlim(-11, 11)
ax.set_ylim(-11, 11)
ax.set_title('минимизация с помощью метода Хука-Дживса')







plt.grid()
plt.tight_layout()
plt.show()
