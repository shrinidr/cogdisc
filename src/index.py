# dissonance_sim.py
import numpy as np
from core_model import BayesianBelief
import copy

def simulate_dissonance_sequence(strategy="honest", steps=10):
    belief = BayesianBelief(alpha=5, beta_=5)
    belief.record_state()

    for step in range(steps):
        outcome = np.random.choice([0, 1], p=[0.8, 0.2])  # mostly conflicting
        pre_update = copy.deepcopy(belief)

        # Apply strategy
        if strategy == "honest":
            belief.observe(outcome)
        elif strategy == "denial" and outcome == 0:
            continue  # skip update
        elif strategy == "confirmation_bias":
            if outcome == 1:
                belief.observe(outcome)
        elif strategy == "soft_update":
            belief.observe(outcome, weight=0.5)
        
        kl = belief.kl_divergence(pre_update)
        print(f"Step {step}: Outcome={outcome}, KL={kl:.4f}")

    belief.plot_history()


if __name__ == "__main__":
    simulate_dissonance_sequence(strategy="honest")
