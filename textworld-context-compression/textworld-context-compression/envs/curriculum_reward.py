"""Potential-based reward shaping for the curriculum (proposal Section 3.6).

Implements

    R_shaped(s_t, s_{t+1}, e) = R_native(s_t, s_{t+1}) + alpha_e * [gamma * Phi(s_{t+1}) - Phi(s_t)]
    alpha_e = alpha_0 * exp(-kappa * e)

with Phi(s) = -shortest_path_distance(room(s), goal_room) on the game's room-adjacency
graph, Phi(terminal) = 0 by definition. Ng, Harada & Russell (1999) show this leaves the
optimal policy unchanged for any potential Phi, so decaying alpha_e to 0 recovers the
original sparse objective by the end of training — the curriculum is a training-time
crutch only, and eval-time reward is always the raw sparse signal (Section 3.6: "At
evaluation time on the crafting distribution, no shaping is applied").

This is deliberately environment-agnostic: it only needs a room-adjacency graph and a
"which room is the agent in" function, both of which the env adapter supplies.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, Iterable, Optional


class PotentialShaper:
    """Graph-distance potential + exponentially-decaying curriculum coefficient.

    Usage:
        shaper = PotentialShaper(alpha_0=1.0, kappa=0.05)
        shaper.set_graph(adjacency, goal_room="kitchen")
        ...
        shaped_r = shaper.shape(native_reward, room_before, room_after, epoch, gamma=0.99)
    """

    def __init__(self, alpha_0: float = 1.0, kappa: float = 0.05, enabled: bool = True):
        self.alpha_0 = float(alpha_0)
        self.kappa = float(kappa)
        self.enabled = bool(enabled)
        self._dist_from_goal: Dict[Hashable, int] = {}
        self._goal_room: Optional[Hashable] = None

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------
    def set_graph(self, adjacency: Dict[Hashable, Iterable[Hashable]], goal_room: Hashable) -> None:
        """Precompute shortest-path distances to `goal_room` via BFS.

        `adjacency` maps room -> iterable of neighbouring rooms (undirected is fine;
        TextWorld door connections are typically reversible). Unreachable rooms get a
        large finite distance rather than infinity, so Phi stays a real number.
        """
        self._goal_room = goal_room
        dist: Dict[Hashable, int] = {goal_room: 0}
        frontier = deque([goal_room])
        while frontier:
            node = frontier.popleft()
            for nbr in adjacency.get(node, ()):
                if nbr not in dist:
                    dist[nbr] = dist[node] + 1
                    frontier.append(nbr)
        self._dist_from_goal = dist
        # Fallback distance for rooms BFS never reached (disconnected subgraph, or a
        # room discovered after the graph snapshot was taken): 2x the max known
        # distance + 1, so it still shrinks monotonically as the agent gets "closer"
        # to any known room, without ever looking better than a truly close room.
        self._fallback_dist = 2 * max(dist.values(), default=0) + 1

    def potential(self, room: Optional[Hashable]) -> float:
        """Phi(s) = -distance(room(s), goal). Phi(terminal) = 0 by construction
        (the goal room has distance 0)."""
        if room is None:
            return -float(self._fallback_dist)
        return -float(self._dist_from_goal.get(room, self._fallback_dist))

    # ------------------------------------------------------------------
    # Curriculum coefficient
    # ------------------------------------------------------------------
    def alpha(self, epoch: float) -> float:
        if not self.enabled:
            return 0.0
        import math

        return self.alpha_0 * math.exp(-self.kappa * epoch)

    # ------------------------------------------------------------------
    # Shaping
    # ------------------------------------------------------------------
    def shape(
        self,
        native_reward: float,
        room_before: Optional[Hashable],
        room_after: Optional[Hashable],
        epoch: float,
        gamma: float = 0.99,
        terminal: bool = False,
    ) -> float:
        """Return R_native + alpha_e * [gamma * Phi(s') - Phi(s)].

        Pass `terminal=True` on the step that ends the episode so Phi(s') is forced
        to 0 regardless of the room-distance lookup (Section 3.6: "The terminal-state
        condition Phi(s_terminal) = 0 is enforced by definition").
        """
        alpha_e = self.alpha(epoch)
        if alpha_e == 0.0:
            return native_reward
        phi_s = self.potential(room_before)
        phi_s_next = 0.0 if terminal else self.potential(room_after)
        return native_reward + alpha_e * (gamma * phi_s_next - phi_s)
