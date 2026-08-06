import pygame
from env.constants import (
    TERRAIN_FOREST, TERRAIN_BRUSH, TERRAIN_WATER, TERRAIN_ROCK,
    FIRE_NONE, FIRE_LOW, FIRE_MID, FIRE_HIGH, FIRE_ASH,
)
from render.asset_manager import TILE_SIZE

TERRAIN_BASE_COLORS = {
    TERRAIN_FOREST: (34, 90, 34),
    TERRAIN_BRUSH:  (150, 130, 40),
    TERRAIN_WATER:  (40, 90, 180),
    TERRAIN_ROCK:   (120, 120, 120),
}

TERRAIN_DECO_KEY = {
    TERRAIN_FOREST: "forest_deco",
    TERRAIN_BRUSH:  "brush_deco",
    TERRAIN_ROCK:   "rock_deco",
    # water has no decoration — flat color only
}

# map fire state -> fire sheet frame index (0..6)
FIRE_STATE_TO_FRAME = {
    FIRE_LOW: 1,
    FIRE_MID: 3,
    FIRE_HIGH: 6,
}

ASH_OVERLAY_COLOR = (50, 50, 50, 160)

# how fast the agent's on-screen position catches up to its logical grid position
# (pixels per second)
AGENT_MOVE_SPEED = TILE_SIZE * 6


class Renderer:
    def __init__(self, assets):
        self.assets = assets
        self.agent_pixel_pos = None  # set on first draw call

    def draw(self, screen, env, dt):
        screen.fill((10, 10, 10))
        self._draw_terrain_and_fire(screen, env)
        self._draw_survivors(screen, env)
        self._draw_agent(screen, env, dt)

    def _draw_terrain_and_fire(self, screen, env):
        for y in range(env.terrain.shape[0]):
            for x in range(env.terrain.shape[1]):
                px, py = x * TILE_SIZE, y * TILE_SIZE
                terrain_type = int(env.terrain[y, x])

                # base color
                color = TERRAIN_BASE_COLORS[terrain_type]
                pygame.draw.rect(screen, color, (px, py, TILE_SIZE, TILE_SIZE))

                # decorative sprite centered on the cell (skip for water)
                deco_key = TERRAIN_DECO_KEY.get(terrain_type)
                if deco_key is not None:
                    sprite = self.assets.get(deco_key)
                    rect = sprite.get_rect(center=(px + TILE_SIZE // 2, py + TILE_SIZE // 2))
                    screen.blit(sprite, rect)

                # fire overlay
                fire_state = int(env.fire_state[y, x])
                if fire_state in FIRE_STATE_TO_FRAME:
                    frame = self.assets.get_fire_frame(FIRE_STATE_TO_FRAME[fire_state])
                    screen.blit(frame, (px, py))
                elif fire_state == FIRE_ASH:
                    overlay = pygame.Surface((TILE_SIZE, TILE_SIZE), pygame.SRCALPHA)
                    overlay.fill(ASH_OVERLAY_COLOR)
                    screen.blit(overlay, (px, py))

                pygame.draw.rect(screen, (0, 0, 0), (px, py, TILE_SIZE, TILE_SIZE), 1)

    def _draw_survivors(self, screen, env):
        sprite = self.assets.get("survivor")
        for i, (sx, sy) in enumerate(env.survivor_positions):
            if env.survivor_rescued[i] or env.survivor_burned[i]:
                continue
            center = (sx * TILE_SIZE + TILE_SIZE // 2, sy * TILE_SIZE + TILE_SIZE // 2)
            rect = sprite.get_rect(center=center)
            screen.blit(sprite, rect)

    def _draw_agent(self, screen, env, dt):
        target_x = env.agent_pos[0] * TILE_SIZE + TILE_SIZE // 2
        target_y = env.agent_pos[1] * TILE_SIZE + TILE_SIZE // 2

        if self.agent_pixel_pos is None:
            self.agent_pixel_pos = [float(target_x), float(target_y)]

        # smoothly move the drawn position toward the logical target
        max_step = AGENT_MOVE_SPEED * dt
        dx = target_x - self.agent_pixel_pos[0]
        dy = target_y - self.agent_pixel_pos[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist <= max_step or dist == 0:
            self.agent_pixel_pos = [float(target_x), float(target_y)]
        else:
            self.agent_pixel_pos[0] += max_step * dx / dist
            self.agent_pixel_pos[1] += max_step * dy / dist

        sprite = self.assets.get("drone")
        rect = sprite.get_rect(center=(int(self.agent_pixel_pos[0]), int(self.agent_pixel_pos[1])))
        screen.blit(sprite, rect)