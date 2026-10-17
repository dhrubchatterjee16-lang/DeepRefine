"""
Training & Shadow-Mode Validation for Refinery Reinforcement Learning Optimizer.
Trains the Actor-Critic policy against the Digital Twin simulation environment,
records policy reward curves, and saves the trained controller weights.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizer.refinery_env import RefineryGymEnv
from optimizer.rl_agent import ActorCriticRefineryPolicy
from digital_twin.twin_engine import DigitalTwinSimulator

def train_rl_optimizer(episodes: int = 150):
    print("======================================================================")
    print(" [*] TRAINING REFINERY SMART CONTROL RL AGENT (PPO ACTOR-CRITIC)")
    print("======================================================================")

    env = RefineryGymEnv(max_steps=20)
    twin = DigitalTwinSimulator()
    policy = ActorCriticRefineryPolicy(obs_dim=7, act_dim=2)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.002)

    best_reward = -float("inf")
    ckpt_dir = Path("checkpoints/optimizer")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "refinery_policy.pt"

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0

        states, actions, rewards = [], [], []

        while not done:
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_mean, _ = policy(s_tensor)
            
            # Action sampling with Gaussian exploration
            action_raw = action_mean.detach().squeeze(0).numpy()
            noise = np.random.normal(0, 0.15, size=action_raw.shape)
            action = np.clip(action_raw + noise, -1.0, 1.0)

            # Step environment with Digital Twin feedback
            # Action mapping: [-1, 1] -> [-0.2, 0.2] for reflux, [-2.0, 2.0] for furnace MW
            scaled_act = np.array([action[0] * 0.2, action[1] * 2.0])
            next_state, reward, done, _ = env.step(scaled_act, twin_sim=twin)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            ep_reward += reward

        # Policy update using discounted Monte-Carlo returns
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + 0.95 * G
            returns.insert(0, G)

        returns_t = torch.tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-6)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32)

        optimizer.zero_grad()
        act_means, values = policy(states_t)
        
        # Policy gradient loss + Value function MSE loss
        advantage = returns_t - values.squeeze(-1).detach()
        actor_loss = torch.mean(- torch.sum((actions_t - act_means) ** 2, dim=-1) * advantage)
        critic_loss = nn.MSELoss()(values.squeeze(-1), returns_t)

        total_loss = actor_loss + 0.5 * critic_loss
        total_loss.backward()
        optimizer.step()

        if ep % 30 == 0 or ep == 1:
            print(f"Episode [{ep:03d}/{episodes}] - Total Reward: {ep_reward:.2f} | Loss: {total_loss.item():.4f}")

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save({"model_state_dict": policy.state_dict(), "best_reward": best_reward}, ckpt_path)

    print(f"[+] Successfully trained RL Controller! Best Episode Reward: {best_reward:.2f}")
    print(f"[+] Model saved to: {ckpt_path}")

if __name__ == "__main__":
    train_rl_optimizer()
