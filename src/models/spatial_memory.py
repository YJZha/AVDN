import torch
import torch.nn as nn
import numpy as np


class SpatialMemory(nn.Module):
    """
    GPS-indexed spatial memory grid.

    Stores running-mean visual features indexed by 2-D grid cells derived
    from GPS coordinates.  At each navigation step the agent writes its
    current frame feature into the matching cell and reads a 3x3 spatial
    neighbourhood to obtain a memory context vector.  This context is
    projected to `out_dim` (= lang_cls_dim = 49) and added as a residual
    to lang_cls, shifting the visual-attention query in ET toward
    "what did I see near here before".

    The key information this provides that ET's temporal Transformer cannot
    efficiently derive: O(1) lookup of whether the current GPS region was
    visited before, and what it looked like.  This is most useful on long
    trajectories where revisited locations are temporally far apart.
    """

    def __init__(self, grid_h=16, grid_w=16, feat_dim=512, out_dim=49,
                 radius_deg=0.003):
        super().__init__()
        self.H = grid_h
        self.W = grid_w
        self.feat_dim = feat_dim
        self.radius_deg = radius_deg   # half-side of the square coverage area

        # 3x3 neighbourhood -> out_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 9, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
        )

        # buffers allocated at reset(), not registered as parameters
        self.grid_feat  = None   # [B, H, W, feat_dim]
        self.grid_count = None   # [B, H, W]

    # ------------------------------------------------------------------ #
    #  lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, batch_size: int, device: torch.device):
        """Call once at the start of every rollout."""
        self.grid_feat  = torch.zeros(batch_size, self.H, self.W,
                                      self.feat_dim, device=device)
        self.grid_count = torch.zeros(batch_size, self.H, self.W,
                                      device=device)

    # ------------------------------------------------------------------ #
    #  coordinate helpers                                                  #
    # ------------------------------------------------------------------ #

    def _gps_to_cell(self, gps_pos: np.ndarray, origin_gps: np.ndarray):
        """
        Map a GPS position to grid indices (row, col).
        Both inputs are (lat, lng) in degrees.
        Returns (row, col) clipped to [0, H-1] x [0, W-1].
        """
        delta = gps_pos - origin_gps                # (2,)
        row = int(np.clip(
            (delta[0] / (2 * self.radius_deg) + 0.5) * self.H,
            0, self.H - 1))
        col = int(np.clip(
            (delta[1] / (2 * self.radius_deg) + 0.5) * self.W,
            0, self.W - 1))
        return row, col

    # ------------------------------------------------------------------ #
    #  read / write                                                        #
    # ------------------------------------------------------------------ #

    def write(self, batch_idx: int, gps_pos: np.ndarray,
              origin_gps: np.ndarray, feat: torch.Tensor):
        """
        Update the cell for `gps_pos` with a running mean of `feat`.
        feat: [feat_dim]  (already on the correct device)
        """
        r, c = self._gps_to_cell(gps_pos, origin_gps)
        n = self.grid_count[batch_idx, r, c]
        self.grid_feat[batch_idx, r, c] = (
            self.grid_feat[batch_idx, r, c] * n + feat
        ) / (n + 1)
        self.grid_count[batch_idx, r, c] += 1

    def read_neighbourhood(self, batch_idx: int, gps_pos: np.ndarray,
                           origin_gps: np.ndarray) -> torch.Tensor:
        """
        Return the 3x3 neighbourhood around `gps_pos` as [9 * feat_dim].
        Cells outside the grid boundary are returned as zero vectors.
        """
        r, c = self._gps_to_cell(gps_pos, origin_gps)
        patches = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.H and 0 <= cc < self.W:
                    patches.append(self.grid_feat[batch_idx, rr, cc])
                else:
                    patches.append(
                        torch.zeros(self.feat_dim,
                                    device=self.grid_feat.device))
        return torch.cat(patches, dim=0)   # [9 * feat_dim]

    # ------------------------------------------------------------------ #
    #  forward: batch write + read + project                               #
    # ------------------------------------------------------------------ #

    def forward(self, im_feats: torch.Tensor,
                current_centers: np.ndarray,
                origin_centers: np.ndarray,
                ended: np.ndarray) -> torch.Tensor:
        """
        im_feats:        [B, feat_dim, 49]  raw frame features
        current_centers: [B, 2] GPS (lat, lng) of current view centre
        origin_centers:  [B, 2] GPS of step-0 view centre (fixed per traj)
        ended:           [B]    bool, skip write/read for finished episodes

        Returns mem_delta [B, out_dim] to be added to lang_cls.
        """
        B = im_feats.shape[0]
        device = im_feats.device

        # spatial mean of frame feature: [B, feat_dim]
        feat_mean = im_feats.mean(dim=-1)   # [B, 512]

        neighbourhoods = []
        for i in range(B):
            # write current frame into grid
            if not ended[i]:
                self.write(i, current_centers[i], origin_centers[i],
                           feat_mean[i].detach())

            # read 3x3 neighbourhood (detached — grid is a running buffer,
            # not part of the computation graph for grad stability)
            nb = self.read_neighbourhood(i, current_centers[i],
                                         origin_centers[i])
            neighbourhoods.append(nb)

        nb_tensor = torch.stack(neighbourhoods, dim=0)   # [B, 9*feat_dim]
        mem_delta = self.proj(nb_tensor)                  # [B, out_dim]
        return mem_delta
