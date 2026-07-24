"""Deterministic round-robin selection for the first FedAvg baseline."""

from __future__ import annotations


class RoundRobinScheduler:
    def __init__(self, num_clients: int, clients_per_round: int) -> None:
        if not 0 < clients_per_round <= num_clients:
            raise ValueError("clients_per_round must be between 1 and num_clients")
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.cursor = 0

    def select(self) -> list[int]:
        selected = [
            (self.cursor + offset) % self.num_clients
            for offset in range(self.clients_per_round)
        ]
        self.cursor = (self.cursor + self.clients_per_round) % self.num_clients
        return selected

