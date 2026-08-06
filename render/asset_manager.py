import pygame
import os

# Paths relative to project root — run scripts with `python -m ...` from root
TILE_SIZE = 48

ASSET_PATHS = {
    "drone": "assets/agents/drone.png",
    "survivor": "assets/characters/tent_smallClosed_NW.png",
    "forest_deco": "assets/tiles/plant_bushDetailed_NE.png",
    "brush_deco": "assets/tiles/plant_bushSmall_NE.png",
    "rock_deco": "assets/tiles/cliff_rock_SW.png",
    "fire_sheet": "assets/fire/ARW 2D Flame Sprite Sheet.png",
}

FONT_PATH = "assets/fonts/Kenney Future Narrow.ttf"

FIRE_FRAME_COUNT = 7
FIRE_FRAME_SIZE = 24  # each frame in the sheet is 24x24 px


class AssetManager:
    def __init__(self):
        self.images = {}  # name -> pygame.Surface
        self.fire_frames = []  # list of pygame.Surface, low->high intensity
        self.font = None

    def load_all(self):
        # decorative/object sprites: crop transparent padding, then scale to fit a cell
        for name in ("drone", "survivor", "forest_deco", "brush_deco", "rock_deco"):
            path = ASSET_PATHS[name]
            self.images[name] = self._load_cropped_scaled(path, max_fraction=0.8)

        self._load_fire_frames(ASSET_PATHS["fire_sheet"])
        self.font = pygame.font.Font(FONT_PATH, 18)

    def _load_cropped_scaled(self, path, max_fraction):
        # loads an image, trims transparent padding, scales to fit within
        # max_fraction of TILE_SIZE while preserving aspect ratio
        raw = pygame.image.load(path).convert_alpha()

        bbox = raw.get_bounding_rect()  # smallest rect containing non-transparent pixels
        cropped = pygame.Surface((bbox.width, bbox.height), pygame.SRCALPHA)
        cropped.blit(raw, (0, 0), bbox)

        max_size = int(TILE_SIZE * max_fraction)
        scale = min(max_size / bbox.width, max_size / bbox.height)
        new_w = max(1, int(bbox.width * scale))
        new_h = max(1, int(bbox.height * scale))

        return pygame.transform.smoothscale(cropped, (new_w, new_h))

    def _load_fire_frames(self, path):
        sheet = pygame.image.load(path).convert_alpha()
        for i in range(FIRE_FRAME_COUNT):
            frame = pygame.Surface((FIRE_FRAME_SIZE, FIRE_FRAME_SIZE), pygame.SRCALPHA)
            frame.blit(sheet, (0, 0), (i * FIRE_FRAME_SIZE, 0, FIRE_FRAME_SIZE, FIRE_FRAME_SIZE))
            scaled = pygame.transform.smoothscale(frame, (TILE_SIZE, TILE_SIZE))
            self.fire_frames.append(scaled)

    def get(self, name):
        return self.images[name]

    def get_fire_frame(self, intensity_index):
        # intensity_index: 0 (lowest) .. FIRE_FRAME_COUNT-1 (highest)
        idx = max(0, min(FIRE_FRAME_COUNT - 1, intensity_index))
        return self.fire_frames[idx]