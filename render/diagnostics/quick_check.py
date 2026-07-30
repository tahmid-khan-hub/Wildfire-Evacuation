import pygame
import sys
from env.emberpath_env import EmberPathEnv
from env.constants import (
    TERRAIN_FOREST, TERRAIN_BRUSH, TERRAIN_WATER, TERRAIN_ROCK,
    FIRE_NONE, FIRE_LOW, FIRE_MID, FIRE_HIGH, FIRE_ASH,
)

TILE_SIZE = 48

TERRAIN_COLORS = {
    TERRAIN_FOREST: (34, 90, 34),
    TERRAIN_BRUSH:  (150, 130, 40),
    TERRAIN_WATER:  (40, 90, 180),
    TERRAIN_ROCK:   (120, 120, 120),
}

FIRE_COLORS = {
    FIRE_NONE: None,
    FIRE_LOW:  (255, 200, 0, 120),
    FIRE_MID:  (255, 120, 0, 160),
    FIRE_HIGH: (200, 0, 0, 200),
    FIRE_ASH:  (60, 60, 60, 180),
}

AGENT_COLOR = (0, 255, 255)
SURVIVOR_COLOR = (255, 0, 255)


def draw_grid(screen, env):
    screen.fill((10, 10, 10))

    for y in range(env.terrain.shape[0]):
        for x in range(env.terrain.shape[1]):
            rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)

            # base terrain
            color = TERRAIN_COLORS[int(env.terrain[y, x])]
            pygame.draw.rect(screen, color, rect)

            # fire overlay, if any
            fire_color = FIRE_COLORS[int(env.fire_state[y, x])]
            if fire_color is not None:
                overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                overlay.fill(fire_color)
                screen.blit(overlay, rect.topleft)

            pygame.draw.rect(screen, (0, 0, 0), rect, 1)  # grid lines

    # survivors (draw before agent so agent is always visible on top)
    for i, (sx, sy) in enumerate(env.survivor_positions):
        if env.survivor_rescued[i] or env.survivor_burned[i]:
            continue
        center = (sx * TILE_SIZE + TILE_SIZE // 2, sy * TILE_SIZE + TILE_SIZE // 2)
        pygame.draw.circle(screen, SURVIVOR_COLOR, center, TILE_SIZE // 4)

    # agent
    ax, ay = env.agent_pos
    center = (ax * TILE_SIZE + TILE_SIZE // 2, ay * TILE_SIZE + TILE_SIZE // 2)
    pygame.draw.circle(screen, AGENT_COLOR, center, TILE_SIZE // 3)


def main():
    env = EmberPathEnv(seed=1)
    env.reset()

    pygame.init()
    width = env.terrain.shape[1] * TILE_SIZE
    height = env.terrain.shape[0] * TILE_SIZE
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("EmberPath - quick check")
    clock = pygame.time.Clock()

    running = True
    step_delay = 0  # frames since last env step
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        step_delay += 1
        if step_delay >= 30:  # advance the env every ~1 sec at 30fps
            step_delay = 0
            action = env.action_space.sample()
            _, reward, terminated, truncated, info = env.step(action)
            print(f"step={info['step_count']} reward={reward:.2f} "
                  f"rescued={info['rescued']} burned={info['burned']}")
            if terminated or truncated:
                print("Episode ended, resetting.")
                env.reset()

        draw_grid(screen, env)
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()