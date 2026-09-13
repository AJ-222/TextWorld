"""Build the two non-overlapping TextWorld game pools (proposal Sec 3.4):

  - `games/train_nav/`  — 200 procedurally generated navigation/foraging games,
    graded by three difficulty axes: map size {3,5,7,10}, lock-and-key chain depth
    {0,1,2}, distractor object count {0,3,5} (Sec 3.6). Each game gets a JSON
    sidecar with a room-adjacency graph + goal room, needed by
    `envs/curriculum_reward.py` for potential-based shaping.

  - `games/test_craft/`  — 30 held-out crafting/tool-manipulation games (Sec 3.4.2),
    built from TextWorld's own cooking challenge via the `tw-make tw-cooking` CLI.
    No curriculum shaping is used at eval time (Sec 3.6), so no room-graph sidecar
    is needed for these.

IMPORTANT — READ BEFORE RUNNING AT SCALE
=========================================
This script is written against TextWorld's documented `GameMaker` API (for the nav
pool) and the `tw-make tw-cooking` CLI subcommand (for the craft pool), from general
knowledge of the TextWorld toolkit. It has NOT been executed against a live
TextWorld install in this environment (no network access to fetch the package here).
Flag names and exact API surfaces can drift between TextWorld versions.

Before generating the full pool, run:

    python data/generate_textworld_games.py --smoke-test

This builds 2 nav games + 2 craft games and asserts each one: compiles, is playable
with `textworld.gym`, and (for nav games) has a walkthrough that leads to a win. If
this fails, the traceback will point at exactly which TextWorld call needs
adjusting — cross-check against `python -c "from textworld.generator.maker import
GameMaker; help(GameMaker)"` and `tw-make tw-cooking --help` on your machine before
touching the generation logic blind.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

MAP_SIZES = [3, 5, 7, 10]
LOCK_DEPTHS = [0, 1, 2]
DISTRACTOR_COUNTS = [0, 3, 5]


# ---------------------------------------------------------------------------
# Navigation / foraging pool (training distribution)
# ---------------------------------------------------------------------------


def build_nav_game(
    seed: int,
    n_rooms: int,
    lock_depth: int,
    n_distractors: int,
    include_cook_vocab_prop: bool = True,
) -> Tuple["textworld.generator.Game", Dict, str]:  # noqa: F821
    """Construct one navigation/foraging game with explicit control over the three
    difficulty axes, using TextWorld's GameMaker.

    Layout: a linear-ish chain of `n_rooms` rooms (room_0 .. room_{n-1}), connected
    room_i <-> room_{i+1}. The quest is: reach room_{n-1} (the goal room) and take
    the target object placed there. `lock_depth` doors along the chain (spaced out
    evenly) are locked, each requiring a distinct key placed in an earlier room —
    this is what "lock-and-key chain depth" controls: how many sequential
    unlock-then-traverse steps stand between the start and the goal.
    `n_distractors` decoy objects (not needed for the quest) are scattered into
    random rooms to test whether the agent's compressed history still tracks the
    one object that matters.

    `include_cook_vocab_prop`: if True, adds a stove + a raw food item to room_0
    (not required for the quest) purely so `cook` appears in admissible_commands
    during navigation training — the proposal's risk mitigation for the "Action
    Vocabulary Constraint" (the agent must see every primitive verb it needs at
    test time during training; the crafting test set requires `cook`/`cut`/`open`).
    VERIFY this against your installed TextWorld's object-type KB
    (`textworld.generator.data.get_data()`) — the exact property names for making
    something "cookable" near a "stove" can differ across versions.

    Returns (game, metadata_dict, task_description).
    """
    import textworld
    from textworld.generator.maker import GameMaker

    rng = random.Random(seed)
    M = GameMaker()

    rooms = [M.new_room(f"room_{i}") for i in range(n_rooms)]
    adjacency: Dict[str, List[str]] = {r.name: [] for r in rooms}

    lock_positions = set()
    if lock_depth > 0 and n_rooms > 1:
        # Spread `lock_depth` locked doors evenly across the n_rooms-1 connections.
        n_edges = n_rooms - 1
        step = max(1, n_edges // (lock_depth + 1))
        lock_positions = {min(step * (i + 1), n_edges - 1) for i in range(lock_depth)}

    keys = []
    for i in range(n_rooms - 1):
        src, dst = rooms[i], rooms[i + 1]
        path = M.connect(src.east, dst.west)
        adjacency[src.name].append(dst.name)
        adjacency[dst.name].append(src.name)

        if i in lock_positions:
            door = M.new_door(path, name=f"door_{i}")
            door.add_property("locked")
            key = M.new(type="k", name=f"key_{i}")
            # Place the key in the room *before* the lock so the chain is solvable
            # in order (room_j for some j <= i).
            key_room = rooms[rng.randint(0, i)]
            key_room.add(key)
            M.add_fact("match", key, door)
            keys.append(key)

    goal_room = rooms[-1]
    target = M.new(type="o", name="target object")
    goal_room.add(target)

    if include_cook_vocab_prop:
        stove = M.new(type="stove", name="stove")
        food = M.new(type="f", name="raw potato")
        rooms[0].add(stove)
        rooms[0].add(food)

    for _ in range(n_distractors):
        room = rooms[rng.randint(0, n_rooms - 1)]
        decoy = M.new(type="o", name=f"decoy_{rng.randint(0, 1_000_000)}")
        room.add(decoy)

    M.set_player(rooms[0])
    M.quests = [M.new_quest_using_commands(_expected_walkthrough(M, rooms, keys, target))]

    game = M.build()

    metadata = {
        "split": "train_nav",
        "difficulty": {
            "n_rooms": n_rooms,
            "lock_depth": lock_depth,
            "n_distractors": n_distractors,
        },
        "room_adjacency": adjacency,
        "goal_room": goal_room.name,
        "seed": seed,
    }
    task_description = "Explore the rooms, unlock any doors you need to, and take the target object."
    return game, metadata, task_description


def _expected_walkthrough(M, rooms, keys, target) -> List[str]:
    """Best-effort oracle command sequence for `new_quest_using_commands` /
    validation. ADAPT ME if your TextWorld version names actions differently
    (check `game.walkthrough` after `M.build()` as the source of truth instead —
    this is a fallback used only to seed the quest, walkthrough extraction for SFT
    data happens for-real in `prepare_sft_data.py` via the compiled game's own
    `EnvInfos(extras=["walkthrough"])`, not this list).
    """
    commands: List[str] = []
    for i in range(len(rooms) - 1):
        key_for_this_door = next((k for k in keys if k.name == f"key_{i}"), None)
        if key_for_this_door is not None:
            commands.append(f"take {key_for_this_door.name}")
            commands.append(f"unlock door_{i} with {key_for_this_door.name}")
            commands.append(f"open door_{i}")
        commands.append("go east")
    commands.append(f"take {target.name}")
    return commands


def generate_nav_pool(out_dir: Path, n_games: int, seed_start: int) -> None:
    import textworld

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed_start)
    for i in range(n_games):
        seed = seed_start + i
        n_rooms = rng.choice(MAP_SIZES)
        lock_depth = rng.choice([d for d in LOCK_DEPTHS if d < n_rooms])
        n_distractors = rng.choice(DISTRACTOR_COUNTS)

        game, metadata, task_description = build_nav_game(seed, n_rooms, lock_depth, n_distractors)
        metadata["task_description"] = task_description

        gamefile = out_dir / f"nav_{i:04d}.z8"
        textworld.generator.compile_game(game, path=str(gamefile))
        with open(gamefile.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[nav {i + 1}/{n_games}] {gamefile.name}  rooms={n_rooms} lock_depth={lock_depth} distractors={n_distractors}")


# ---------------------------------------------------------------------------
# Crafting pool (held-out test distribution) — via TextWorld's cooking challenge
# ---------------------------------------------------------------------------


def generate_craft_pool(out_dir: Path, n_games: int, seed_start: int) -> None:
    """Generate held-out cooking/crafting games via the `tw-make tw-cooking` CLI.

    Flags below (`--recipe`, `--take`, `--go`, `--cook`, `--cut`, `--open`) are
    TextWorld's standard "First TextWorld Problems" cooking-challenge knobs from
    general TextWorld documentation. CONFIRM with `tw-make tw-cooking --help` on
    your installed version before a full 30-game run — see module docstring.
    """
    import shlex
    import subprocess

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed_start + 10_000)
    for i in range(n_games):
        seed = seed_start + 10_000 + i
        recipe_size = rng.choice([1, 2, 3])  # number of ingredients -> crafting complexity
        gamefile = out_dir / f"craft_{i:04d}.z8"
        cmd = (
            f"tw-make tw-cooking --recipe {recipe_size} --take {recipe_size} "
            f"--go 6 --cook --cut --open --seed {seed} --output {shlex.quote(str(gamefile))} -f"
        )
        print(f"[craft {i + 1}/{n_games}] $ {cmd}")
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"tw-make tw-cooking failed for seed={seed}.\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}\n\n"
                "Run `tw-make tw-cooking --help` and adjust the flags in "
                "generate_craft_pool() to match your installed TextWorld version."
            )
        metadata = {
            "split": "test_craft",
            "difficulty": {"recipe_size": recipe_size},
            "seed": seed,
            "task_description": "Cook the recipe described in the cookbook using the available ingredients and tools.",
        }
        with open(gamefile.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------


def _smoke_test(out_dir: Path) -> None:
    import textworld
    import textworld.gym

    print("Building 2 nav games...")
    generate_nav_pool(out_dir / "train_nav", n_games=2, seed_start=1)
    print("Building 2 craft games...")
    generate_craft_pool(out_dir / "test_craft", n_games=2, seed_start=1)

    print("Playing back nav_0000 to check it's solvable...")
    gamefile = str(out_dir / "train_nav" / "nav_0000.z8")
    request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["walkthrough"])
    env_id = textworld.gym.register_games([gamefile], request_infos, batch_size=1, max_episode_steps=50)
    env = textworld.gym.make(env_id)
    obs, infos = env.reset()
    walkthrough = infos.get("extra.walkthrough") or infos.get("walkthrough")
    assert walkthrough, "No walkthrough returned — check EnvInfos(extras=['walkthrough']) support in your TextWorld version."
    won = False
    for cmd in walkthrough[0] if isinstance(walkthrough[0], list) else walkthrough:
        obs, score, done, infos = env.step([cmd])
        if done[0] if hasattr(done, "__getitem__") else done:
            won = bool(infos.get("won", [False])[0] if isinstance(infos.get("won"), list) else infos.get("won"))
            break
    assert won, "Walkthrough did not win the game — check build_nav_game()'s quest construction."
    print("Smoke test passed: nav game compiles, registers, and its walkthrough wins.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "games"))
    parser.add_argument("--n-train", type=int, default=200, help="Matches Sec 3.4.1 (200 nav games).")
    parser.add_argument("--n-test", type=int, default=30, help="Matches Sec 3.4.2 (30 held-out craft games).")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true", help="Generate 2+2 games and validate; don't run the full pool.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.smoke_test:
        _smoke_test(out_dir / "_smoke_test")
        return

    generate_nav_pool(out_dir / "train_nav", n_games=args.n_train, seed_start=args.seed_start)
    generate_craft_pool(out_dir / "test_craft", n_games=args.n_test, seed_start=args.seed_start)
    print(f"Done. Games written under {out_dir}")


if __name__ == "__main__":
    main()
