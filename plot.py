import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

A = 0.25
B = "tab:blue"
W = "tab:orange"


def ma(arr, count):
    l = len(arr)
    m = []
    for i in range(count, l):
        j = i - count
        m.append(np.mean(arr[j:i]))
    return np.array(m)


def plot(ax, arr, title, episodes=-1, alpha=A, legend=True):
    ax.set_title(title)
    ax.set_xlim([0, episodes])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    for i in range(2):
        l = "White" if i else "Black"
        c = W if i else B
        ax.plot(arr[i, :episodes], label=l, alpha=alpha, c=c)
    if legend:
        ax.legend()
        ax.grid()
    return ax


def plot_ma(ax, arr, episodes=-1, count: int = 50):
    for i in range(2):
        c = W if i else B
        ax.plot(range(count, episodes), ma(arr[i, :episodes], count), c=c, alpha=1)
    return ax


def bar(ax, arr, title, episodes, alpha=A):
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_xlim([0, lst])
    ax.set_ylabel("Value")

    for i in range(2):
        l = "White" if i else "Black"
        h = arr[i, :episodes]
        x = range(lst)
        ax.bar(x, h, label=l, alpha=alpha)
    ax.legend()
    ax.grid()
    return ax


def plot_moves(ax, moves, episodes, count: int = 50):
    arr = moves.sum(axis=0)[:episodes]
    ax.plot(arr, alpha=A, c=B)
    ax.plot(range(count, episodes), ma(arr, count), alpha=1, c=B)
    ax.set_title("Total Moves")
    ax.set_xlim([0, episodes])
    ax.set_xlabel("Episode")
    ax.grid()


def density(arr, count, episode):
    a = arr.max(axis=0)
    return [np.sum(a[max(0, i - count) : i]) / count for i in range(episode)]


def plot_check_mates(
    ax, check_mates_arr: np.ndarray, episodes: int, count_density: int
):
    #     ax.plot(check_mates_arr.max(axis=0)[:episodes], alpha=0.25)
    density_ax = ax.twinx()
    density_arr = density(check_mates_arr, count_density, episodes)
    density_ax.plot(
        range(episodes),
        density_arr,
        color="tab:green",
        alpha=1,
        label=f"total check mates rate for {count_density} episodes",
        linewidth=2,
    )
    density_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    density_ax.legend()
    density_ax.grid()
    plot(ax, check_mates_arr, "Check Mates", episodes, alpha=0.25, legend=False)


ALPHA = 0.25
COUNT = 512  # 512
for name in ["Double Agents", "Single Agent"]:
    print(name, "...")
    folder = "".join(name.split(" "))
    folder = f"results/{folder}"
    moves = np.load(f"{folder}/moves.npy")
    mates = np.load(f"{folder}/mates_win.npy")
    checks = np.load(f"{folder}/checks_win.npy")
    rewards = np.load(f"{folder}/rewards.npy")
    episodes = np.max(np.where(moves[0] != 0)) + 1

    fig, axs = plt.subplots(2, 2, figsize=(20, 12), dpi=200)
    fig.suptitle(f"{name} | {episodes} Episodes")

    plot(axs[0, 0], rewards, "Rewards", episodes, alpha=ALPHA)
    plot_ma(axs[0, 0], rewards, episodes, count=32)

    plot_moves(axs[0, 1], moves, episodes, count=32)

    plot(axs[1, 0], checks, "Checks", episodes, alpha=ALPHA)
    plot_ma(axs[1, 0], checks, episodes, count=32)

    plot_check_mates(axs[1, 1], mates, episodes, COUNT)

    fig.tight_layout()
    fig.savefig(f"{folder}/plots.jpeg")
