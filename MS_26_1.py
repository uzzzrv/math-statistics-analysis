import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import binom

# Параметры для задания 1
V = 26
N = 200
n_binom = 8
p_binom = 0.552
seed = V + 10  # 36

rng = np.random.default_rng(seed)

# 1. Генерация выборки
sample = rng.binomial(n_binom, p_binom, N)
sorted_sample = np.sort(sample)

# 2. Статистический ряд
unique_vals, counts = np.unique(sample, return_counts=True)
m = len(unique_vals)
w_i = counts / N
s_i = np.cumsum(w_i)

stat_table = [[unique_vals[i], counts[i], w_i[i], s_i[i]] for i in range(m)]

print("=" * 80)
print("Статистический ряд (биномиальное распределение, вариант 26)")
print("=" * 80)
print(tabulate(stat_table, headers=["x*_i", "n_i", "w_i", "s_i"], floatfmt=".5f"))
print()

# 3. Выборочные характеристики
x_bar = np.sum(unique_vals * w_i)
mu2 = np.sum((unique_vals ** 2) * w_i)
D_B = mu2 - x_bar ** 2
sigma_B = np.sqrt(D_B)

mu3 = np.sum((unique_vals ** 3) * w_i)
mu4 = np.sum((unique_vals ** 4) * w_i)
mu3_cent = mu3 - 3 * mu2 * x_bar + 2 * (x_bar ** 3)
mu4_cent = mu4 - 4 * mu3 * x_bar + 6 * mu2 * (x_bar ** 2) - 3 * (x_bar ** 4)
gamma1 = mu3_cent / (sigma_B ** 3)
gamma2 = mu4_cent / (sigma_B ** 4) - 3

# Мода
max_count = np.max(counts)
modes = unique_vals[counts == max_count]
sample_mode = modes[0] if len(modes) == 1 else np.mean(modes)

# Медиана
cumsum = np.cumsum(w_i)
median_idx = np.where(cumsum >= 0.5)[0][0]
sample_median = unique_vals[median_idx]

# Теоретические значения
theor_probs = binom.pmf(unique_vals, n_binom, p_binom)
theor_mean = n_binom * p_binom
theor_var = n_binom * p_binom * (1 - p_binom)
theor_std = np.sqrt(theor_var)

# Теоретическая мода
mode_theor_candidate = int((n_binom + 1) * p_binom)
if (n_binom + 1) * p_binom % 1 == 0:
    theor_mode = mode_theor_candidate - 0.5
else:
    theor_mode = mode_theor_candidate

theor_median = binom.ppf(0.5, n_binom, p_binom)

q = 1 - p_binom
theor_gamma1 = (q - p_binom) / np.sqrt(n_binom * p_binom * q)
theor_gamma2 = (1 - 6 * p_binom * q) / (n_binom * p_binom * q)

# Таблица сравнения характеристик (5 знаков)
chars_table = [
    ["Среднее значение", f"{x_bar:.5f}", f"{theor_mean:.5f}", f"{abs(x_bar - theor_mean):.5f}",
     f"{abs(x_bar - theor_mean) / theor_mean:.5f}" if theor_mean != 0 else "-"],
    ["Дисперсия", f"{D_B:.5f}", f"{theor_var:.5f}", f"{abs(D_B - theor_var):.5f}",
     f"{abs(D_B - theor_var) / theor_var:.5f}" if theor_var != 0 else "-"],
    ["Ср. кв. отклонение", f"{sigma_B:.5f}", f"{theor_std:.5f}", f"{abs(sigma_B - theor_std):.5f}",
     f"{abs(sigma_B - theor_std) / theor_std:.5f}" if theor_std != 0 else "-"],
    ["Мода", f"{sample_mode:.5f}", f"{theor_mode:.5f}", f"{abs(sample_mode - theor_mode):.5f}",
     f"{abs(sample_mode - theor_mode) / theor_mode:.5f}" if theor_mode != 0 else "-"],
    ["Медиана", f"{sample_median:.5f}", f"{theor_median:.5f}", f"{abs(sample_median - theor_median):.5f}",
     f"{abs(sample_median - theor_median) / theor_median:.5f}" if theor_median != 0 else "-"],
    ["Коэф. асимметрии", f"{gamma1:.5f}", f"{theor_gamma1:.5f}", f"{abs(gamma1 - theor_gamma1):.5f}",
     f"{abs(gamma1 - theor_gamma1) / theor_gamma1:.5f}" if theor_gamma1 != 0 else "-"],
    ["Коэф. эксцесса", f"{gamma2:.5f}", f"{theor_gamma2:.5f}", f"{abs(gamma2 - theor_gamma2):.5f}",
     f"{abs(gamma2 - theor_gamma2) / theor_gamma2:.5f}" if theor_gamma2 != 0 else "-"]
]

print("Таблица сравнения характеристик")
print("=" * 80)
print(tabulate(chars_table, headers=["Показатель", "Выборочное", "Теоретическое",
                                     "Абс. откл.", "Отн. откл."], floatfmt="s"))
print()

# Таблица сравнения относительных частот и теоретических вероятностей (5 знаков)
comp_table = []
for i in range(m):
    comp_table.append([unique_vals[i], f"{w_i[i]:.5f}", f"{theor_probs[i]:.5f}", f"{abs(w_i[i] - theor_probs[i]):.5f}"])

delta_max = max(abs(w_i - theor_probs))

# Последняя строка
comp_table.append(["Σ", f"{np.sum(w_i):.5f}", f"{np.sum(theor_probs):.5f}", f"Δ_max = {delta_max:.5f}"])

print("Таблица сравнения w_i и p̃_i")
print("=" * 80)
print(tabulate(comp_table, headers=["x*_i", "w_i", "p̃_i", "|w_i - p̃_i|"], floatfmt="s"))
print()

# 4. График эмпирической функции распределения
x_ecdf = np.sort(sample)
y_ecdf = np.arange(1, N + 1) / N

plt.figure(figsize=(10, 6))

plt.hlines(0, 1, x_ecdf[0], color='black')

for i in range(N):
    if i == 0:
        plt.hlines(y_ecdf[i], x_ecdf[i], x_ecdf[i + 1], color='black')
    elif i == N - 1:
        plt.hlines(y_ecdf[i], x_ecdf[i], x_ecdf[i], color='black')
    else:
        plt.hlines(y_ecdf[i], x_ecdf[i], x_ecdf[i + 1], color='black')

plt.xlabel('x')
plt.ylabel('F_N(x)')
plt.title(f'Эмпирическая функция распределения (биномиальное, n={n_binom}, p={p_binom})')
plt.xlim(0.5, 9)
plt.ylim(-0.05, 1.05)
plt.grid(alpha=0.3)
plt.show()

# 5. Полигон относительных частот
x_all = np.arange(0, n_binom + 1)
theor_probs_all = binom.pmf(x_all, n_binom, p_binom)

plt.figure(figsize=(10, 6))
plt.plot(unique_vals, w_i, 'bo-', label='Эмпирические частоты')
plt.plot(x_all, theor_probs_all, 'ro-', color='red', label='Теоретические вероятности')
plt.xlabel('x')
plt.ylabel('Относительная частота / Вероятность')
plt.title(f'Полигон относительных частот (биномиальное, n={n_binom}, p={p_binom})')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Вывод выборки для отчёта
print("Выборка (200 значений, 20x10):")
sample_2d = sample.reshape(20, 10)
for row in sample_2d:
    print(" ".join(f"{val:3d}" for val in row))

print("\nУпорядоченная выборка (20x10):")
sorted_2d = sorted_sample.reshape(20, 10)
for row in sorted_2d:
    print(" ".join(f"{val:3d}" for val in row))