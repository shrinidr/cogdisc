# core_model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, entropy

class BayesianBelief:
    def __init__(self, alpha=5, beta_=5, name="Competence"):
        self.alpha = alpha
        self.beta = beta_
        self.name = name
        self.history = []

    def prior(self):
        return beta(self.alpha, self.beta)

    def observe(self, outcome, weight=1.0):
        """Update based on outcome (1 is positive evidence and 0 is neg)"""
        if outcome == 1:
            self.alpha += weight
        else:
            self.beta += weight
        self.record_state()

    def kl_divergence(self, other):
        """KL divergence to another Beta distribution"""
        p = beta(self.alpha, self.beta)
        q = beta(other.alpha, other.beta)
        xs = np.linspace(0.01, 0.99, 100)
        return entropy(p.pdf(xs), q.pdf(xs))

    def record_state(self):
        self.history.append((self.alpha, self.beta))

    def plot_history(self):
        save_path = "belief_evol.png"
        for i, (a, b) in enumerate(self.history):
            dist = beta(a, b)
            xs = np.linspace(0, 1, 100)
            plt.plot(xs, dist.pdf(xs), label=f"Step {i}")
        plt.title(f"Belief evolution: {self.name}")
        plt.xlabel("Belief strength")
        plt.ylabel("Density")
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        print(f"[✔] Plot saved to {save_path}")
