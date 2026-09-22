"""Plot the FRGB3D reliability model as a function of range and reflectivity.

Standalone script (no src/ dependency) reproducing the reliability-vs-range
curves from the paper's reliability model figure.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def calculate_reliability_for_curve(distances, reflectivity_percent):
    """
    Compute the reliability-vs-range curve for a fixed reflectivity value.

    Args:
        distances: array of range values (meters).
        reflectivity_percent: reflectivity, 0-100.

    Returns:
        Array of reliability values in [0, 1].
    """
    min_std = 0.5
    a_low = (3 - min_std) / (90.0 * 90.0)      # coefficient at 10% reflectivity
    a_high = (0.8 - min_std) / (90.0 * 90.0)   # coefficient at 90% reflectivity
    refl_factor = (a_low - a_high) / (90.0 - 10.0)

    a_at_0_percent = a_low + 10.0 * refl_factor
    a_at_100_percent = max(0.0, a_high - 10.0 * refl_factor)

    refl = reflectivity_percent
    if refl == 0.0:
        a = a_at_0_percent
    elif 0 < refl < 10.0:
        t = refl / 10.0
        a = a_at_0_percent * (1.0 - t) + a_low * t
    elif refl == 10.0:
        a = a_low
    elif 10 < refl < 90.0:
        a = a_low - (refl - 10.0) * refl_factor
    elif refl == 90.0:
        a = a_high
    elif 90 < refl < 100.0:
        t = (refl - 90.0) / 10.0
        a = a_high * (1.0 - t) + a_at_100_percent * t
    else:
        a = a_at_100_percent
    a = max(0.0, a)

    reliability = np.zeros_like(distances, dtype=np.float32)
    for i, r in enumerate(distances):
        precision_std = a * (r * r) + min_std
        reliability[i] = max(0.0, 1.0 - 0.01 * precision_std)
    return reliability


def main():
    distances = np.linspace(0.5, 200, 201)
    reflectivity_values = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
    labels = [f'Refl {v}%' for v in reflectivity_values]

    plt.figure(figsize=(12, 8))
    for i, refl in enumerate(reflectivity_values):
        reliability = calculate_reliability_for_curve(distances, refl)
        plt.plot(distances, reliability, color=colors[i], linewidth=2, label=labels[i])

    plt.title('Reliability vs Distance for Different Reflectivity', fontsize=16)
    plt.xlabel('Range (m)', fontsize=14)
    plt.ylabel('Reliability', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.ylim(0.8, 1.01)

    plt.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)

    special_distances = [1, 90, 200]
    for dist in special_distances:
        for i, refl in enumerate(reflectivity_values):
            rel = calculate_reliability_for_curve(np.array([dist]), refl)[0]
            plt.plot(dist, rel, 'o', color=colors[i], markersize=6)
            if dist != 1:
                plt.annotate(f'{rel:.3f}', (dist, rel), xytext=(dist + 5, rel),
                             fontsize=9, color=colors[i])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
