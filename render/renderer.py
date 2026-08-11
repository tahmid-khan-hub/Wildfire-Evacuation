import pygame
from env.constants import (
    TERRAIN_FOREST, TERRAIN_BRUSH, TERRAIN_WATER, TERRAIN_ROCK,
    FIRE_LOW, FIRE_MID, FIRE_HIGH, FIRE_ASH,
)
from render.asset_manager import TILE_SIZE

WATER_COLOR = (40, 90, 180)

FIRE_STATE_TO_FRAME = {
    FIRE_LOW: 1,
    FIRE_MID: 3,
    FIRE_HIGH: 6,
}

ASH_OVERLAY_COLOR = (50, 50, 50, 160)
AGENT_MOVE_SPEED = TILE_SIZE * 6
AGENT_ANIM_FPS_DIVISOR = 6  # lower = faster animation cycling


class Renderer:
    def __init__(self, assets):
        self.assets = assets
        self.agent_pixel_pos = None
        self.frame_counter = 0

    def draw(self, screen, env, dt):
        self.frame_counter += 1
        screen.fill((10, 10, 10))
        self._draw_terrain_and_fire(screen, env)
        self._draw_survivors(screen, env)
        self._draw_agent(screen, env, dt)

    def _draw_terrain_and_fire(self, screen, env):
        # cells to keep clear of bush decoration so characters stay visible
        occupied_cells = {env.agent_pos}
        for i, pos in enumerate(env.survivor_positions):
            if not env.survivor_rescued[i] and not env.survivor_burned[i]:
                occupied_cells.add(pos)

        for y in range(env.terrain.shape[0]):
            for x in range(env.terrain.shape[1]):
                px, py = x * TILE_SIZE, y * TILE_SIZE
                terrain_type = int(env.terrain[y, x])
                cell_occupied = (x, y) in occupied_cells

                if terrain_type == TERRAIN_WATER:
                    pygame.draw.rect(screen, WATER_COLOR, (px, py, TILE_SIZE, TILE_SIZE))
                else:
                    # every non-water terrain sits on the same green base tile
                    screen.blit(self.assets.get("forest_tile"), (px, py))

                    if terrain_type == TERRAIN_ROCK:
                        sprite = self.assets.get("rock_deco")
                        rect = sprite.get_rect(center=(px + TILE_SIZE // 2, py + TILE_SIZE // 2))
                        screen.blit(sprite, rect)
                    elif terrain_type == TERRAIN_BRUSH and not cell_occupied:
                        deco_key = "bush_large" if (x + y) % 2 == 0 else "bush_small"
                        sprite = self.assets.get(deco_key)
                        rect = sprite.get_rect(center=(px + TILE_SIZE // 2, py + TILE_SIZE // 2))
                        screen.blit(sprite, rect)
                    elif terrain_type == TERRAIN_FOREST and not cell_occupied:
                        # visual-only: ~80% of forest tiles get a bush drawn
                        # over them. Deterministic per-cell hash so it's
                        # stable across frames. Does NOT touch env.terrain,
                        # so fire spread / ignition probabilities are
                        # unchanged and no retraining is needed.
                        cell_hash = (x * 92821 + y * 68917) % 100
                        if cell_hash < 80:
                            deco_key = "bush_large" if cell_hash % 2 == 0 else "bush_small"
                            sprite = self.assets.get(deco_key)
                            rect = sprite.get_rect(center=(px + TILE_SIZE // 2, py + TILE_SIZE // 2))
                            screen.blit(sprite, rect)

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
        for i, (sx, sy) in enumerate(env.survivor_positions):
            if env.survivor_rescued[i] or env.survivor_burned[i]:
                continue
            sprite = self.assets.get_survivor_sprite(i)
            center = (sx * TILE_SIZE + TILE_SIZE // 2, sy * TILE_SIZE + TILE_SIZE // 2)
            rect = sprite.get_rect(center=center)
            screen.blit(sprite, rect)

    def _draw_agent(self, screen, env, dt):
        target_x = env.agent_pos[0] * TILE_SIZE + TILE_SIZE // 2
        target_y = env.agent_pos[1] * TILE_SIZE + TILE_SIZE // 2

        if self.agent_pixel_pos is None:
            self.agent_pixel_pos = [float(target_x), float(target_y)]

        max_step = AGENT_MOVE_SPEED * dt
        dx = target_x - self.agent_pixel_pos[0]
        dy = target_y - self.agent_pixel_pos[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist <= max_step or dist == 0:
            self.agent_pixel_pos = [float(target_x), float(target_y)]
        else:
            self.agent_pixel_pos[0] += max_step * dx / dist
            self.agent_pixel_pos[1] += max_step * dy / dist

        frame_index = self.frame_counter // AGENT_ANIM_FPS_DIVISOR
        sprite = self.assets.get_agent_frame(frame_index)
        rect = sprite.get_rect(center=(int(self.agent_pixel_pos[0]), int(self.agent_pixel_pos[1])))
        screen.blit(sprite, rect)