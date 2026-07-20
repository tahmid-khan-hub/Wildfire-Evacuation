from env.emberpath_env import EmberPathEnv

env = EmberPathEnv(seed=1)
obs, info = env.reset()
print("Reset OK. Agent at:", env.agent_pos)

for i in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"step {i}: reward={reward:.2f}, rescued={info['rescued']}, burned={info['burned']}")
    if terminated or truncated:
        print("Episode ended.")
        break

print("No crashes. Env works.")