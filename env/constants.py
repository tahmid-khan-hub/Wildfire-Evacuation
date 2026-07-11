# Grid dimensions
GRID_WIDTH = 12
GRID_HEIGHT = 10
MAX_EPISODE_STEPS = 200

# Terrain types (static, set at grid generation)
TERRAIN_FOREST = 0 # dense forest — slow ignition, high fuel (burns long)
TERRAIN_BRUSH = 1 # dry brush — fast ignition, burns fast
TERRAIN_WATER = 2 # fireproof, impassable to agent
TERRAIN_ROCK = 3 # fireproof, passable (firebreak / safe corridor)

TERRAIN_NAMES = {
    TERRAIN_FOREST: "forest",
    TERRAIN_BRUSH: "brush",
    TERRAIN_WATER: "water",
    TERRAIN_ROCK: "rock",
}

# Terrain that the agent physically can not enter
IMPASSBLE_TERRAIN = {TERRAIN_WATER}

# Terrain that can never catch fire
FIREPROOF_TERRAIN = {TERRAIN_WATER, TERRAIN_ROCK}

# Base ignition probability per adjacent burning neighbor, per step
IGNITION_PROB = {
    TERRAIN_FOREST: 0.10,
    TERRAIN_BRUSH: 0.35,
    TERRAIN_WATER: 0.0,
    TERRAIN_ROCK: 0.0,
}

# How many steps a cell stays "burning" (intensity ramps) before becoming ash
BURN_DURATION = {
    TERRAIN_FOREST: 6, # high fuel, burns long
    TERRAIN_BRUSH: 3, # burns fast, done quick
    TERRAIN_WATER: 0,
    TERRAIN_ROCK: 0,
}

# Fire states (it is dynamic, chnages over episode)
FIRE_NONE = 0
FIRE_LOW = 1
FIRE_MID = 2
FIRE_HIGH = 3
FIRE_ASH = 4

FIRE_STATE_NAMES = {
    FIRE_NONE: "none",
    FIRE_LOW: "low",
    FIRE_MID: "mid",
    FIRE_HIGH: "high",
    FIRE_ASH: "ash",
}

# Cells the agent cannot safely enter
DANGEROUS_FIRE_STATES = {FIRE_LOW, FIRE_MID, FIRE_HIGH}

# Actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}

ACTION_DELTAS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Rewards
REWARD_STEP_PENALTY = -0.1 # small cost per step, encourages efficiency
REWARD_RESCUE = 50.0 # per survivor successfully evacuated
REWARD_SURVIVOR_BURNED = -30.0 # per survivor lost to fire
REWARD_AGENT_BURNED = -100.0 # agent enters a burning cell -> episode ends
REWARD_DISTANCE_SHAPING = 0.5 # scaled by reduction in distance to nearest survivor
REWARD_TIMEOUT = -10.0 # episode hits max steps without full rescue

# Survivor config
NUM_SURVIVORS = 3

# Fire ignition config
NUM_INITIAL_IGNITIONS = 1 # how many cells start on fire at reset()
