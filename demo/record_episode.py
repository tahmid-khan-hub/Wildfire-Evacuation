import os
import pygame
import imageio
import numpy as np
from stable_baselines3 import DQN
from env.emberpath_env import EmberPathEnv
from render.asset_manager import AssetManager, TILE_SIZE
from render.renderer import Renderer
from render.hud import HUD, HUD_HEIGHT

MODEL_PATH = "models/dqn_emberpath.zip"  # swap to models/ppo_emberpath.zip or Q-table for other demos
SEED = 42
OUTPUT_PATH = "demo/output/dqn_episode.gif"
FRAME_DURATION = 0.9  # seconds each frame displays in the gif
FINAL_FRAME_HOLD = 2.5

def surface_to_array(surface):
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2))  # pygame is (w,h,c) -> imageio wants (h,w,c)

def main():
    os.makedirs("demo/output", exist_ok=True)

    env = EmberPathEnv(seed=SEED)
    obs, info = env.reset(seed=SEED)
    model = DQN.load(MODEL_PATH)

    pygame.init()
    grid_w = env.terrain.shape[1] * TILE_SIZE
    grid_h = env.terrain.shape[0] * TILE_SIZE
    screen = pygame.display.set_mode((grid_w, grid_h + HUD_HEIGHT), flags=pygame.HIDDEN)  # off-screen, no window needed

    assets = AssetManager()
    assets.load_all()
    renderer = Renderer(assets)
    hud = HUD(assets)

    frames = []

    # dt=999 forces the agent sprite to snap straight to its grid position each
    # frame (skips the smooth in-between animation) — simpler and more reliable
    # for a frame-by-frame GIF capture than real-time interpolation.
    renderer.draw(screen.subsurface((0, HUD_HEIGHT, grid_w, grid_h)), env, dt=999)
    hud.draw(screen, info, 0.0, grid_w)
    frames.append(surface_to_array(screen))

    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))

        renderer.draw(screen.subsurface((0, HUD_HEIGHT, grid_w, grid_h)), env, dt=999)
        hud.draw(screen, info, reward, grid_w)
        frames.append(surface_to_array(screen))

    durations = [FRAME_DURATION] * (len(frames) - 1) + [FINAL_FRAME_HOLD]
    imageio.mimsave(OUTPUT_PATH, frames, duration=durations)
    print(f"Saved GIF to {OUTPUT_PATH} ({len(frames)} frames)")
    print(f"Final: rescued={info['rescued']}/3, burned={info['burned']}/3")


if __name__ == "__main__":
    main()