import hashlib
import numpy as np
from logging import INFO
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg
from flwr.common import ArrayRecord


class VerifiableRobustStrategy(FedAvg):
    """
    FedAvg extended with three security layers:
      1. Commit-reveal verification  — mitigates Type 1 tampering attacks
      2. Trimmed mean aggregation    — mitigates Type 2 label/data poisoning
      3. In-memory Merkle ledger     — computes 32-byte state roots for on-chain posting
    """

    def __init__(self, beta=0.2, malicious_fraction=0.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.malicious_fraction = malicious_fraction
        self.ledger = {}    # round → {"root": hex, "n_verified": int, "n_rejected": int}

    # New signature: (server_round, arrays, config, grid)
    # arrays = current global ArrayRecord
    # config = ConfigRecord sent to clients
    # grid   = handles node sampling and message delivery
    def configure_train(self, server_round, arrays, config, grid):
        config["server_round"] = str(server_round)
        config["malicious_fraction"] = str(self.malicious_fraction)
        return super().configure_train(server_round, arrays, config, grid)

    # New signature: (server_round, replies)
    # replies = list of Message objects, one per responding client
    # Each Message carries an ArrayRecord (weights) + MetricRecord (metrics)
    def aggregate_train(self, server_round, replies):
        replies     = list(replies)
        verified    = []
        rejected    = []
        commitments = []

        for msg in replies:
            if not msg.has_content():
                continue

            weights = msg.content[self.arrayrecord_key].to_numpy_ndarrays()
            metrics = msg.content["metrics"]
            n       = int(metrics["num-examples"])
            logloss = float(metrics.get("train_logloss", 0.0))     # ← capture this

            commit_info = msg.content.get("commit_info", {})
            commit_h    = str(commit_info.get("commitment", ""))
            nonce_h     = str(commit_info.get("nonce", ""))

            if commit_h and nonce_h:
                nonce        = bytes.fromhex(nonce_h)
                weight_bytes = b''.join(w.tobytes() for w in weights)
                round_bytes  = server_round.to_bytes(4, 'big')
                recomputed   = hashlib.sha256(weight_bytes + nonce + round_bytes).hexdigest()

                if recomputed == commit_h:
                    log(INFO, f"  R{server_round} | VERIFIED  | n={n}")
                    verified.append((weights, n, logloss))
                    commitments.append(bytes.fromhex(commit_h))
                else:
                    log(INFO, f"  R{server_round} | REJECTED  | commitment mismatch (Type 1 caught)")
                    rejected.append("mismatch")
            else:
                log(INFO, f"  R{server_round} | REJECTED  | no commitment provided")
                rejected.append("no_commit")

        root = self._merkle_root(commitments) if commitments else b'\x00' * 32
        self.ledger[server_round] = {
            "root": root.hex(), "n_verified": len(verified), "n_rejected": len(rejected),
        }
        log(INFO, f"  R{server_round} | root={root.hex()[:16]}... | "
                f"verified={len(verified)} rejected={len(rejected)}")

        if not verified:
            return None, {}

        arrays_list = [w for w, _, _ in verified]
        aggregated  = self._trimmed_mean(arrays_list, self.beta)

        # ── weighted-average train_logloss across verified clients only ──────
        total_n     = sum(n for _, n, _ in verified)
        agg_logloss = sum(lo * n for _, n, lo in verified) / total_n if total_n else 0.0

        return ArrayRecord.from_numpy_ndarrays(aggregated), {"train_logloss": agg_logloss}

    # ── Helpers ───────────────────────────────────────────────────────────

    def _trimmed_mean(self, arrays_list, beta):
        """
        At each coordinate: sort across clients, remove k from each end, average the rest.
        k = int(beta * n_clients)
        """
        n = len(arrays_list)
        k = int(beta * n)
        result = []
        for i in range(len(arrays_list[0])):
            stacked  = np.stack([c[i] for c in arrays_list], axis=0)
            sorted_v = np.sort(stacked, axis=0)
            trimmed  = sorted_v[k: n - k] if k > 0 and (n - k) > k else sorted_v
            result.append(np.mean(trimmed, axis=0))
        return result

    def _merkle_root(self, leaves):
        """
        Build a binary Merkle tree over commitment bytes.
        Returns one 32-byte root — the only value that needs to go on-chain.
        """
        if not leaves:
            return b'\x00' * 32
        layer = [hashlib.sha256(leaf).digest() for leaf in leaves]
        while len(layer) > 1:
            if len(layer) % 2 == 1:
                layer.append(layer[-1])
            layer = [
                hashlib.sha256(layer[i] + layer[i + 1]).digest()
                for i in range(0, len(layer), 2)
            ]
        return layer[0]

    def print_ledger(self):
        """Call this after training completes to inspect the simulated on-chain checkpoints."""
        log(INFO, "\n── In-memory ledger (simulates on-chain checkpoints) ──")
        for r, entry in self.ledger.items():
            log(INFO,
                f"  Round {r}: root={entry['root'][:16]}... "
                f"verified={entry['n_verified']} rejected={entry['n_rejected']}")
                