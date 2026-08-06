import pygame

HUD_HEIGHT = 40
HUD_BG_COLOR = (15, 15, 15)
HUD_TEXT_COLOR = (240, 240, 240)


class HUD:
    def __init__(self, assets):
        self.font = assets.font

    def draw(self, screen, info, reward, grid_width_px):
        rect = pygame.Rect(0, 0, grid_width_px, HUD_HEIGHT)
        pygame.draw.rect(screen, HUD_BG_COLOR, rect)

        text = (
            f"Step {info['step_count']}   "
            f"Rescued {info['rescued']}   "
            f"Burned {info['burned']}   "
            f"Reward {reward:.2f}"
        )
        surf = self.font.render(text, True, HUD_TEXT_COLOR)
        screen.blit(surf, (10, HUD_HEIGHT // 2 - surf.get_height() // 2))