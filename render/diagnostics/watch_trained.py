import pygame
import sys
from env.emberpath_env import EmberPathEnv
from agents.q_learning import QLearningAgent, discretize_state
from render.asset_manager import AssetManager, TILE_SIZE
from render.renderer import Renderer
from render.hud import HUD, HUD_HEIGHT

Q_TABLE_PATH = "models/q_table.pkl"
SEED = 42


def main():
    env = EmberPathEnv(seed=SEED)
    env.reset()

    agent = QLearningAgent()
    agent.load(Q_TABLE_PATH)
    agent.epsilon = 0.0

    pygame.init()
    grid_w = env.terrain.shape[1] * TILE_SIZE
    grid_h = env.terrain.shape[0] * TILE_SIZE
    screen = pygame.display.set_mode((grid_w, grid_h + HUD_HEIGHT))
    pygame.display.set_caption("EmberPath - trained Q-learning agent")
    clock = pygame.time.Clock()

    assets = AssetManager()
    assets.load_all()
    renderer = Renderer(assets)
    hud = HUD(assets)

    running = True
    step_delay = 0.0
    last_reward = 0.0
    last_info = env._get_info()

    while running:
        dt = clock.tick(60) / 1000.0  # seconds since last frame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        step_delay += dt
        if step_delay >= 0.5:  # one env step every 0.5s
            step_delay = 0.0
            state = discretize_state(env)
            action = agent.select_action(state, greedy=True)
            _, reward, terminated, truncated, info = env.step(action)
            last_reward, last_info = reward, info
            print(f"step={info['step_count']} reward={reward:.2f} "
                  f"rescued={info['rescued']} burned={info['burned']}")
            if terminated or truncated:
                print(f"Episode ended. Final: rescued={info['rescued']}, burned={info['burned']}")
                running = False

        renderer.draw(screen.subsurface((0, HUD_HEIGHT, grid_w, grid_h)), env, dt)
        hud.draw(screen, last_info, last_reward, grid_w)
        pygame.display.flip()

    pygame.time.wait(2000)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()