import pygame

TILE_SIZE = 48

ASSET_PATHS = {
    "drone_walk_sheet": "assets/agents/Walk.png",  # 4-frame strip, 118x268 per frame

    "forest_tile": "assets/tiles/forest.png",       # base tile under every non-water terrain

    "rock_deco": "assets/tiles/Rock4_1.png",
    "bush_large": "assets/tiles/Bush_orange_flowers1.png",
    "bush_small": "assets/tiles/Bush_orange_flowers3.png",

    "survivor_boy": "assets/characters/Boy.png",
    "survivor_man": "assets/characters/Man.png",
    "survivor_woman": "assets/characters/Woman.png",
    "survivor_boy_hurt": "assets/characters/Boy_hurt.png",
    "survivor_man_hurt": "assets/characters/Man_hurt.png",
    "survivor_woman_hurt": "assets/characters/Woman_hurt.png",

    "fire_sheet": "assets/fire/ARW 2D Flame Sprite Sheet.png",
}

FONT_PATH = "assets/fonts/Kenney Future Narrow.ttf"

FIRE_FRAME_COUNT = 7
FIRE_FRAME_SIZE = 24

AGENT_FRAME_COUNT = 4
AGENT_FRAME_W = 48
AGENT_FRAME_H = 48

SURVIVOR_CYCLE = ["survivor_boy", "survivor_man", "survivor_woman"]


class AssetManager:
    def __init__(self):
        self.images = {}
        self.fire_frames = []
        self.agent_frames = []
        self.font = None

    def load_all(self):
        self.images["forest_tile"] = self._load_fullsize(ASSET_PATHS["forest_tile"])

        self.images["rock_deco"] = self._load_cropped_scaled(ASSET_PATHS["rock_deco"], max_fraction=0.75)
        self.images["bush_large"] = self._load_cropped_scaled(ASSET_PATHS["bush_large"], max_fraction=0.95)
        self.images["bush_small"] = self._load_cropped_scaled(ASSET_PATHS["bush_small"], max_fraction=0.55)

        for name in ("survivor_boy", "survivor_man", "survivor_woman",
                     "survivor_boy_hurt", "survivor_man_hurt", "survivor_woman_hurt"):
            self.images[name] = self._load_cropped_scaled(ASSET_PATHS[name], max_fraction=0.8)

        self._load_fire_frames(ASSET_PATHS["fire_sheet"])
        self._load_agent_frames(ASSET_PATHS["drone_walk_sheet"])

        self.font = pygame.font.Font(FONT_PATH, 18)

    def _load_fullsize(self, path):
        raw = pygame.image.load(path).convert_alpha()
        return pygame.transform.smoothscale(raw, (TILE_SIZE, TILE_SIZE))

    def _load_cropped_scaled(self, path, max_fraction):
        raw = pygame.image.load(path).convert_alpha()
        bbox = raw.get_bounding_rect()
        cropped = pygame.Surface((bbox.width, bbox.height), pygame.SRCALPHA)
        cropped.blit(raw, (0, 0), bbox)

        max_size = int(TILE_SIZE * max_fraction)
        scale = min(max_size / bbox.width, max_size / bbox.height)
        new_w = max(1, int(bbox.width * scale))
        new_h = max(1, int(bbox.height * scale))
        return pygame.transform.smoothscale(cropped, (new_w, new_h))

    def _slice_strip(self, path, frame_count, frame_w, frame_h):
        sheet = pygame.image.load(path).convert_alpha()
        frames = []
        for i in range(frame_count):
            frame = pygame.Surface((frame_w, frame_h), pygame.SRCALPHA)
            frame.blit(sheet, (0, 0), (i * frame_w, 0, frame_w, frame_h))
            frames.append(frame)
        return frames

    def _load_fire_frames(self, path):
        frames = self._slice_strip(path, FIRE_FRAME_COUNT, FIRE_FRAME_SIZE, FIRE_FRAME_SIZE)
        self.fire_frames = [pygame.transform.smoothscale(f, (TILE_SIZE, TILE_SIZE)) for f in frames]

    def _load_agent_frames(self, path):
        frames = self._slice_strip(path, AGENT_FRAME_COUNT, AGENT_FRAME_W, AGENT_FRAME_H)
        scaled = []
        for f in frames:
            bbox = f.get_bounding_rect()
            cropped = pygame.Surface((bbox.width, bbox.height), pygame.SRCALPHA)
            cropped.blit(f, (0, 0), bbox)
            max_size = int(TILE_SIZE * 0.85)
            scale = min(max_size / bbox.width, max_size / bbox.height)
            new_w = max(1, int(bbox.width * scale))
            new_h = max(1, int(bbox.height * scale))
            scaled.append(pygame.transform.smoothscale(cropped, (new_w, new_h)))
        self.agent_frames = scaled

    def get(self, name):
        return self.images[name]

    def get_survivor_sprite(self, index, hurt=False):
        key = SURVIVOR_CYCLE[index % len(SURVIVOR_CYCLE)]
        if hurt:
            key += "_hurt"
        return self.images[key]

    def get_fire_frame(self, intensity_index):
        idx = max(0, min(FIRE_FRAME_COUNT - 1, intensity_index))
        return self.fire_frames[idx]

    def get_agent_frame(self, frame_index):
        return self.agent_frames[frame_index % len(self.agent_frames)]