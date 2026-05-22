import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import expon

# Параметры для задания 3
V = 26
N = 200
lambda_exp = 1.078
seed = V + 10

rng = np.random.default_rng(seed)

# 1. Генерация выборки
sample = rng.exponential(scale=1/lambda_exp, size=N)
sorted_sample = np.sort(sample)

# 2. Интервальный ряд по формуле Стерджеса
m = 1 + int(np.log2(N))
a0 = 0
am = max(sample)
h = (am - a0) / m

bounds = [a0 + i * h for i in range(m + 1)]
bounds[0] = a0
bounds[-1] = am

intervals = []
n_i = []
for i in range(m):
    left = bounds[i]
    right = bounds[i + 1]
    if i == 0:
        count = np.sum((sample >= left) & (sample <= right))
    else:
        count = np.sum((sample > left) & (sample <= right))
    n_i.append(count)
    intervals.append(f"[{left:.4f}, {right:.4f}]" if i == 0 else f"({left:.4f}, {right:.4f}]")

w_i = np.array(n_i) / N
midpoints = [(bounds[i] + bounds[i + 1]) / 2 for i in range(m)]

print("=" * 80)
print("Интервальный ряд (показательное распределение, вариант 26)")
print(f"λ = {lambda_exp}, m = {m}, h = {h:.5f}")
print("=" * 80)
print(tabulate([[intervals[i], n_i[i], f"{w_i[i]:.5f}"] for i in range(m)],
               headers=["Интервал", "n_i", "w_i"], floatfmt="s"))
print()

print("Ассоциированный статистический ряд")
print("=" * 80)
print(tabulate([[f"{midpoints[i]:.5f}", n_i[i], f"{w_i[i]:.5f}"] for i in range(m)],
               headers=["x*_i", "n_i", "w_i"], floatfmt="s"))
print()

# 3. Выборочные характеристики
x_bar = np.sum(midpoints * w_i)
mu2 = np.sum((np.array(midpoints) ** 2) * w_i)
D_B = mu2 - x_bar ** 2

# Поправка Шеппарда
sheppard_correction = h**2 / 12
D_B_sheppard = D_B - sheppard_correction
sigma_B = np.sqrt(D_B_sheppard)

mu3 = np.sum((np.array(midpoints) ** 3) * w_i)
mu4 = np.sum((np.array(midpoints) ** 4) * w_i)
mu3_cent = mu3 - 3 * mu2 * x_bar + 2 * (x_bar ** 3)
mu4_cent = mu4 - 4 * mu3 * x_bar + 6 * mu2 * (x_bar ** 2) - 3 * (x_bar ** 4)
gamma1 = mu3_cent / (sigma_B ** 3)
gamma2 = mu4_cent / (sigma_B ** 4) - 3

# Мода
max_w_idx = np.argmax(w_i)
if max_w_idx == 0:
    w_left, w_right = 0, w_i[1] if m > 1 else 0
elif max_w_idx == m - 1:
    w_left, w_right = w_i[m - 2] if m > 1 else 0, 0
else:
    w_left, w_right = w_i[max_w_idx - 1], w_i[max_w_idx + 1]

w_k = w_i[max_w_idx]
a_left = bounds[max_w_idx]
sample_mode = a_left + h * (w_k - w_left) / (2 * w_k - w_left - w_right)

# Медиана
cumsum_w = np.cumsum(w_i)
median_idx = np.where(cumsum_w >= 0.5)[0][0]
cumsum_prev = cumsum_w[median_idx - 1] if median_idx > 0 else 0
a_left_median = bounds[median_idx]
sample_median = a_left_median + h / w_i[median_idx] * (0.5 - cumsum_prev)

# Теоретические значения
theor_mean = 1 / lambda_exp
theor_var = 1 / (lambda_exp ** 2)
theor_std = 1 / lambda_exp
theor_mode = 0
theor_median = np.log(2) / lambda_exp
theor_gamma1 = 2
theor_gamma2 = 6

# Теоретические вероятности попадания в интервалы
theor_probs = []
for i in range(m):
    left = bounds[i]
    right = bounds[i + 1]
    prob = expon.cdf(right, scale=1/lambda_exp) - expon.cdf(left, scale=1/lambda_exp)
    theor_probs.append(prob)

# Таблица сравнения характеристик (5 знаков)
chars_table = [
    ["Среднее значение", f"{x_bar:.5f}", f"{theor_mean:.5f}", f"{abs(x_bar - theor_mean):.5f}",
     f"{abs(x_bar - theor_mean) / theor_mean:.5f}"],
    ["Дисперсия (с поправкой Шеппарда)", f"{D_B_sheppard:.5f}", f"{theor_var:.5f}", f"{abs(D_B_sheppard - theor_var):.5f}",
     f"{abs(D_B_sheppard - theor_var) / theor_var:.5f}"],
    ["Ср. кв. отклонение", f"{sigma_B:.5f}", f"{theor_std:.5f}", f"{abs(sigma_B - theor_std):.5f}",
     f"{abs(sigma_B - theor_std) / theor_std:.5f}"],
    ["Мода", f"{sample_mode:.5f}", f"{theor_mode:.5f}", f"{abs(sample_mode - theor_mode):.5f}", "-"],
    ["Медиана", f"{sample_median:.5f}", f"{theor_median:.5f}", f"{abs(sample_median - theor_median):.5f}",
     f"{abs(sample_median - theor_median) / theor_median:.5f}"],
    ["Коэф. асимметрии", f"{gamma1:.5f}", f"{theor_gamma1:.5f}", f"{abs(gamma1 - theor_gamma1):.5f}",
     f"{abs(gamma1 - theor_gamma1) / theor_gamma1:.5f}"],
    ["Коэф. эксцесса", f"{gamma2:.5f}", f"{theor_gamma2:.5f}", f"{abs(gamma2 - theor_gamma2):.5f}",
     f"{abs(gamma2 - theor_gamma2) / theor_gamma2:.5f}"]
]

print("Таблица сравнения характеристик")
print("=" * 80)
print(tabulate(chars_table, headers=["Показатель", "Выборочное", "Теоретическое",
                                     "Абс. откл.", "Отн. откл."], floatfmt="s"))
print()

# Таблица сравнения частот с строкой суммы (5 знаков)
comp_table = []
for i in range(m):
    comp_table.append([intervals[i], f"{w_i[i]:.5f}", f"{theor_probs[i]:.5f}", f"{abs(w_i[i] - theor_probs[i]):.5f}"])

delta_max = max(abs(w_i - np.array(theor_probs)))

# Последняя строка
comp_table.append(["Σ", f"{np.sum(w_i):.5f}", f"{np.sum(theor_probs):.5f}", f"Δ_max = {delta_max:.5f}"])

print("Таблица сравнения w_i и p_i")
print("=" * 80)
print(tabulate(comp_table, headers=["Интервал", "w_i", "p_i", "|w_i - p_i|"], floatfmt="s"))
print()

# 4. График эмпирической функции распределения
x_ecdf = np.sort(sample)
y_ecdf = np.arange(1, N + 1) / N
x_min = x_ecdf[0]

plt.figure(figsize=(10, 6))

for i in range(N):
    if i == 0:
        plt.hlines(y_ecdf[i], x_ecdf[i], x_ecdf[i + 1], color='blue')
    elif i == N - 1:
        plt.hlines(y_ecdf[i], x_ecdf[i], x_ecdf[i], color='blue')
    else:
        plt.hlines(y_ecdf[i], x_ecdf[i], x_ecdf[i + 1], color='blue')

plt.xlabel('x')
plt.ylabel('F_N(x)')
plt.title(f'Эмпирическая функция распределения (показательное, λ={lambda_exp})')
plt.xlim(x_min - 0.2, max(sample) + 0.3)
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.3)
plt.show()

# 5. Гистограмма относительных частот с наложенной плотностью
plt.figure(figsize=(10, 6))

plt.bar(midpoints, w_i / h, width=h, alpha=0.7, edgecolor='black',
        label='Гистограмма относительных частот')

x_density = np.linspace(0, am, 1000)
y_density = lambda_exp * np.exp(-lambda_exp * x_density)
plt.plot(x_density, y_density, 'r-', linewidth=2, label='Плотность показательного распределения')

plt.xlabel('x')
plt.ylabel('Плотность')
plt.title(f'Гистограмма относительных частот (показательное, λ={lambda_exp})')
plt.xlim(-0.2, am + 0.1)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Вывод выборки
print("Выборка (200 значений, 20x10):")
sample_2d = sample.reshape(20, 10)
for row in sample_2d:
    print(" ".join(f"{val:.4f}" for val in row))

print("\nУпорядоченная выборка (20x10):")
sorted_2d = sorted_sample.reshape(20, 10)
for row in sorted_2d:
    print(" ".join(f"{val:.4f}" for val in row))