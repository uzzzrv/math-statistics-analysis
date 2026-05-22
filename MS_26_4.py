import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

data_raw = [
    [0.29297, 2.52939, 1.46824, -0.91311, 2.15283, 2.88439, 0.14333, 2.79641, 1.24939, 1.23731],
    [1.48662, 1.61200, 2.12823, 2.17977, 0.49980, 2.33349, 0.28481, 0.25715, 2.20457, 1.69754],
    [0.48142, 1.61273, 3.95596, 1.52854, 0.29930, 0.01067, 2.03068, -1.80032, -0.81144, 3.31097],
    [3.46753, 1.75267, 1.40328, 0.28060, 2.13994, 0.95756, 1.49450, 3.16176, 1.27274, 0.19814],
    [3.38456, 1.74111, 0.70151, 0.73033, 0.15564, -0.06260, 1.10153, 2.49905, 1.31117, 1.00288],
    [1.23869, 1.03933, 0.45502, 2.07833, 3.29605, -0.67064, 1.80775, 0.50576, 1.46707, 1.46474],
    [2.24262, 0.72785, 0.07670, 1.24044, 0.91120, 3.02254, 1.04418, 1.92094, 1.68810, 3.93769],
    [1.74430, 0.14080, -1.58060, -0.05256, 0.21533, -1.25017, 0.53591, -0.17770, 0.10516, 2.19713],
    [0.81551, 0.44374, 1.18311, 1.27238, 3.94528, 1.68154, 2.23172, 2.61693, 3.04902, 0.93606],
    [-0.67949, 0.56858, 0.62813, 2.54421, -1.62739, -1.17117, 0.22135, -0.29432, 1.45307, 1.70093],
    [1.24760, 1.44131, 1.85318, 1.23268, -0.06718, 1.91462, 0.64858, 2.43442, 0.17873, 3.52191],
    [3.14809, -0.49370, -0.28056, 1.41021, -0.30350, 0.12902, 0.39658, 3.53285, 0.91038, -0.82012],
    [1.67231, 0.01443, -0.28683, 2.93860, 0.62856, 0.56782, -0.74073, 1.05680, 2.08788, 2.16556],
    [1.53707, 4.44928, 1.44079, 1.18589, 0.09862, 1.35785, 1.19488, 1.63096, 0.60567, 1.47849],
    [1.33968, 1.13728, -0.33609, 2.37656, 0.58511, 0.98624, -0.19084, -0.78614, 2.96577, 2.12307],
    [1.72222, -0.49281, 1.51393, 0.22457, 0.12953, 0.15185, 1.87941, 0.60179, 2.61087, 1.97677],
    [1.94666, 1.55048, 0.95411, -0.11155, -1.03819, -0.93644, 3.55469, 0.86955, 0.86475, 0.71881],
    [0.12250, -0.85982, 2.12448, 2.15391, 2.09153, 0.47491, 0.81232, 2.73882, 0.97715, 3.30774],
    [1.84768, 1.77912, 3.29284, 2.74530, 0.07495, 1.32814, 0.48055, 0.19215, 2.30258, 2.19628],
    [-0.73106, 0.46539, 1.98620, 2.14910, 3.93130, 1.38261, 0.30510, 1.00808, -0.25161, 2.77682]
]

y = np.array(data_raw).flatten()
N = len(y)

print("Вариант 26")

print("\nВыборка из нормального распределения")
for i, row in enumerate(data_raw):
    print(f"{i + 1:2d} " + " ".join([f"{x:8.5f}" for x in row]))

y_sorted = np.sort(y)
print("\nУпорядоченная выборка из нормального распределения")
for i in range(20):
    row = y_sorted[i * 10:(i + 1) * 10]
    print(f"{i + 1:2d} " + " ".join([f"{x:8.5f}" for x in row]))

a0 = np.min(y)
am = np.max(y)
m = 1 + int(np.floor(np.log2(N)))
h = (am - a0) / m

print(f"\nПараметры интервального ряда")
print(f"a0 = {a0:.5f}, am = {am:.5f}, m = {m}, h = {h:.5f}")

boundaries = [a0 + i * h for i in range(m + 1)]
boundaries[-1] = am

n_k = []
for i in range(m):
    if i == 0:
        count = np.sum((y >= boundaries[i]) & (y <= boundaries[i + 1]))
    else:
        count = np.sum((y > boundaries[i]) & (y <= boundaries[i + 1]))
    n_k.append(count)

w_k = np.array(n_k) / N

print("\nИнтервальный ряд")
print("-" * 70)
print(f"{'k':<3} {'Интервал':<30} {'n_k':<8} {'W_k':<10}")
print("-" * 70)
for i in range(m):
    if i == 0:
        interval = f"[{boundaries[i]:.5f}, {boundaries[i + 1]:.5f}]"
    else:
        interval = f"({boundaries[i]:.5f}, {boundaries[i + 1]:.5f}]"
    print(f"{i + 1:<3} {interval:<30} {n_k[i]:<8} {w_k[i]:<10.5f}")
print("-" * 70)
print(f"{'Σ':<3} {'':<30} {sum(n_k):<8} {sum(w_k):<10.5f}")

a_hat = np.mean(y)
variance_raw = np.var(y, ddof=0)
sigma2_hat = variance_raw - (h ** 2) / 12
sigma_hat = np.sqrt(sigma2_hat)

print(f"\nОценки параметров")
print(f"a = {a_hat:.5f}")
print(f"σ² = {sigma2_hat:.5f}")
print(f"σ = {sigma_hat:.5f}")


def phi0(t):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-t ** 2 / 2)


print("\n Таблица 4.4. Вычисление p_k*")
print(f"{'k':<3} {'a_k':<12} {'(a_k - a)/σ':<18} {'(1/σ)·φ0((a_k - a)/σ)':<28} {'Φ((a_k - a)/σ)':<20} {'p_k*':<10}")

Phi_vals = []
for k in range(m + 1):
    z = (boundaries[k] - a_hat) / sigma_hat
    Phi_vals.append(norm.cdf(z))

z0 = (boundaries[0] - a_hat) / sigma_hat
phi_val0 = phi0(z0)
term0 = (1 / sigma_hat) * phi_val0
print(f"{0:<3} {boundaries[0]:<12.5f} {z0:<18.5f} {term0:<28.5f} {Phi_vals[0]:<20.5f} {'—':<10}")

p_star = []
for k in range(m - 1):
    if k == 0:
        p = Phi_vals[1]
    else:
        p = Phi_vals[k + 1] - Phi_vals[k]
    p_star.append(p)

    z = (boundaries[k + 1] - a_hat) / sigma_hat
    phi_val = phi0(z)
    term = (1 / sigma_hat) * phi_val

    if k == 0:
        interval_label = f"(-∞, {boundaries[1]:.5f}]"
    else:
        interval_label = f"({boundaries[k]:.5f}, {boundaries[k + 1]:.5f}]"

    print(f"{k + 1:<3} {interval_label:<12} {z:<18.5f} {term:<28.5f} {Phi_vals[k + 1]:<20.5f} {p:<10.5f}")

p_last = 1 - sum(p_star)
p_star.append(p_last)

k = m - 1
z = (boundaries[m] - a_hat) / sigma_hat
phi_val = phi0(z)
term = (1 / sigma_hat) * phi_val
interval_label = f"({boundaries[m - 1]:.5f}, +∞)"
print(f"{k + 1:<3} {interval_label:<12} {z:<18.5f} {term:<28.5f} {Phi_vals[m]:<20.5f} {p_last:<10.5f}")

print(f"{'Σ':<3} {'':<12} {'':<18} {'':<28} {'':<20} {sum(p_star):<10.5f}")

chi2_terms = []
for i in range(m):
    if p_star[i] > 0:
        val = N * (w_k[i] - p_star[i]) ** 2 / p_star[i]
    else:
        val = 0
    chi2_terms.append(val)

chi2_B = sum(chi2_terms)
max_diff = max(abs(w_k[i] - p_star[i]) for i in range(m))

print("\n Таблица 4.5. Вычисление выборочного значения критерия")
print(f"{'k':<3} {'Интервал':<30} {'w_k':<12} {'p_k*':<12} {'|w_k - p_k*|':<15} {'N(w_k - p_k*)²/p_k*':<20}")

for i in range(m):
    if i == 0:
        interval = f"[{boundaries[i]:.5f}, {boundaries[i + 1]:.5f}]"
    else:
        interval = f"({boundaries[i]:.5f}, {boundaries[i + 1]:.5f}]"
    print(
        f"{i + 1:<3} {interval:<30} {w_k[i]:<12.5f} {p_star[i]:<12.5f} {abs(w_k[i] - p_star[i]):<15.5f} {chi2_terms[i]:<20.5f}")

print(f"{'Σ':<3} {'':<30} {sum(w_k):<12.5f} {sum(p_star):<12.5f} {max_diff:<15.5f} {chi2_B:<20.5f}")

l = m - 3
alpha = 0.03

crit_table = {4: 10.711898, 5: 12.374618, 6: 13.967617, 7: 15.509090, 8: 17.010493}
chi2_crit = crit_table.get(l, None)

if chi2_crit is None:
    from scipy.stats import chi2

    chi2_crit = chi2.ppf(1 - alpha, l)

print("\nПроверка гипотезы")
print(f"Уровень значимости α = {alpha:.2f}")
print(f"Число степеней свободы l = m - 3 = {l}")
print(f"Критическое значение χ²_кр = {chi2_crit:.5f}")
print(f"Выборочное значение χ²_B = {chi2_B:.5f}")

if chi2_B <= chi2_crit:
    print(f"\nВывод: χ²_B = {chi2_B:.5f} ≤ χ²_кр = {chi2_crit:.5f} → гипотеза НЕ ПРОТИВОРЕЧИТ данным.")
    print(
        f"Нормальное распределение N({a_hat:.5f}, {sigma_hat:.5f}²) соответствует выборке при уровне значимости α = {alpha}.")
else:
    print(f"\nВывод: χ²_B = {chi2_B:.5f} > χ²_кр = {chi2_crit:.5f} → гипотеза ПРОТИВОРЕЧИТ данным.")
    print(
        f"Нормальное распределение N({a_hat:.5f}, {sigma_hat:.5f}²) НЕ соответствует выборке при уровне значимости α = {alpha}.")

plt.figure(figsize=(10, 6))
plt.hist(y, bins=boundaries, density=True, edgecolor='black', alpha=0.7,
         label='Гистограмма относительных частот')
x_grid = np.linspace(a0 - 0.5, am + 0.5, 500)
pdf_normal = norm.pdf(x_grid, a_hat, sigma_hat)
plt.plot(x_grid, pdf_normal, 'r-', linewidth=2,
         label=f'N({a_hat:.5f}, {sigma_hat:.5f}²)')
plt.xlabel('x')
plt.ylabel('Плотность')
plt.title('Рисунок 4.1. Гистограмма относительных частот и график плотности нормального распределения')
plt.legend()
plt.grid(alpha=0.3)
plt.show()