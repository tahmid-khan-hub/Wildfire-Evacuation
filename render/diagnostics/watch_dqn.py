import pygame
import sys
from stable_baselines3 import DQN
from env.emberpath_env import EmberPathEnv
from render.asset_manager import AssetManager, TILE_SIZE
from render.renderer import Renderer
from render.hud import HUD, HUD_HEIGHT

MODEL_PATH = "models/dqn_emberpath.zip"
SEED = 42


def main():
    env = EmberPathEnv(seed=SEED)
    env.reset(seed=SEED)
    model = DQN.load(MODEL_PATH)

    pygame.init()
    grid_w = env.terrain.shape[1] * TILE_SIZE
    grid_h = env.terrain.shape[0] * TILE_SIZE
    screen = pygame.display.set_mode((grid_w, grid_h + HUD_HEIGHT))
    pygame.display.set_caption("EmberPath - trained DQN agent")
    clock = pygame.time.Clock()

    assets = AssetManager()
    assets.load_all()
    renderer = Renderer(assets)
    hud = HUD(assets)

    obs, info = env.reset(seed=SEED)
    running = True
    step_delay = 0.0
    last_reward = 0.0

    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        step_delay += dt
        if step_delay >= 0.5:
            step_delay = 0.0
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            last_reward = reward
            print(f"step={info['step_count']} reward={reward:.2f} "
                  f"rescued={info['rescued']} burned={info['burned']}")
            if terminated or truncated:
                print(f"Episode ended. Final: rescued={info['rescued']}, burned={info['burned']}")
                running = False

        renderer.draw(screen.subsurface((0, HUD_HEIGHT, grid_w, grid_h)), env, dt)
        hud.draw(screen, info, last_reward, grid_w)
        pygame.display.flip()

    pygame.time.wait(2000)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()