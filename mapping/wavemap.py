import numpy as np
import pywavemap as wave
from pywavemap import InterpolationMode
import gc


class WaveMapper:
    def __init__(self, params=None):

        try:
            self.map = wave.Map.create(
                {
                    "type": "hashed_chunked_wavelet_octree",
                    "min_cell_width": {"meters": params["min_cell_width"]},
                }
            )
            self.pipeline = wave.Pipeline(self.map)
            self.pipeline.add_operation(
                {"type": "threshold_map", "once_every": {"seconds": 5.0}}
            )
            self.pipeline.add_integrator(
                "my_integrator",
                {
                    "projection_model": {
                        "type": "pinhole_camera_projector",
                        "width": params["width"],
                        "height": params["height"],
                        "fx": params["fx"],
                        "fy": params["fy"],
                        "cx": params["cx"],
                        "cy": params["cy"],
                    },
                    "measurement_model": {
                        "type": "continuous_ray",
                        "range_sigma": {"meters": 0.05},
                        "scaling_free": 0.2,
                        "scaling_occupied": 0.4,
                    },
                    "integration_method": {
                        "type": "hashed_chunked_wavelet_integrator",
                        "min_range": {"meters": params["min_range"]},
                        "max_range": {"meters": params["max_range"]},
                    },
                },
            )
            self.params = params

            # print all params
            print("WaveMapper initialized with parameters:")
            for key, value in params.items():
                print(f"{key}: {value}")
            self.depth_buffer = []
            self.res = params["resolution"]
            self.query_space = (
                np.mgrid[-20 : 20 : self.res, -20 : 20 : self.res, -5 : 5 : self.res]
                .reshape(3, -1)
                .T
            )
            self.occ_pts = None
            self.free_pts = None
            self.grid_min = -20.0
            self.grid_max = 20.0
            self.grid_size = int(round((self.grid_max - self.grid_min) / self.res))
            # pywavemap's Python API does not expose allocated/observed voxels.
            # This mask is therefore updated only by actual RGB-D rays and is
            # the source of truth for unobserved space in geometry completion.
            self.observed_mask_2d = np.zeros(
                (self.grid_size, self.grid_size), dtype=bool
            )
            self.observed_ray_stride = int(params.get("observed_ray_stride", 12))

        except Exception as e:
            raise RuntimeError(f"Failed to initialize WaveMapper: {e}")

    def get_parameters(self):
        """
        Get the parameters of the mapper.
        Returns a dictionary of parameters.
        """
        return self.params

    def insert_depth_to_buffer(self, depth, transform):
        """
        Insert depth data into the map using the provided transformation.
        transform: 4x4 numpy, camera IN world frame
        """
        pose = wave.Pose(transform)
        image = wave.Image(np.array(depth).transpose())
        self.depth_buffer.append({"pose": pose, "image": image})
        self._update_observed_mask(depth, transform)

    def _update_observed_mask(self, depth, transform):
        """Project sampled, valid depth rays into the accumulated XY observed mask."""
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth image, got {depth.shape}")

        stride = max(self.observed_ray_stride, 1)
        rows = np.arange(0, depth.shape[0], stride)
        cols = np.arange(0, depth.shape[1], stride)
        vv, uu = np.meshgrid(rows, cols, indexing="ij")
        sampled_depth = depth[vv, uu]
        valid = (
            np.isfinite(sampled_depth)
            & (sampled_depth >= float(self.params["min_range"]))
            & (sampled_depth <= float(self.params["max_range"]))
        )
        if not np.any(valid):
            return

        z = sampled_depth[valid]
        u = uu[valid].astype(np.float32)
        v = vv[valid].astype(np.float32)
        camera_points = np.stack(
            (
                (u - float(self.params["cx"])) * z / float(self.params["fx"]),
                (v - float(self.params["cy"])) * z / float(self.params["fy"]),
                z,
                np.ones_like(z),
            ),
            axis=1,
        )
        world_endpoints = (np.asarray(transform) @ camera_points.T).T[:, :3]
        origin = np.asarray(transform, dtype=float)[:3, 3]

        max_samples = int(np.ceil(float(self.params["max_range"]) / self.res)) + 1
        fractions = np.linspace(0.0, 1.0, max_samples, dtype=np.float32)
        ray_xy = origin[None, None, :2] + fractions[None, :, None] * (
            world_endpoints[:, None, :2] - origin[None, None, :2]
        )
        grid = np.floor((ray_xy - self.grid_min) / self.res).astype(np.int32)
        inside = (
            (grid[..., 0] >= 0)
            & (grid[..., 0] < self.grid_size)
            & (grid[..., 1] >= 0)
            & (grid[..., 1] < self.grid_size)
        )
        grid = grid[inside]
        self.observed_mask_2d[grid[:, 0], grid[:, 1]] = True

    def get_navigation_projection(
        self,
        nav_level,
        min_height=0.1,
        max_height=1.3,
        height_step=0.2,
    ):
        """Return a Wavemap-backed 2D navigation slice with explicit unobserved cells."""
        xs = self.grid_min + (np.arange(self.grid_size) + 0.5) * self.res
        ys = self.grid_min + (np.arange(self.grid_size) + 0.5) * self.res
        xx, yy = np.meshgrid(xs, ys, indexing="ij")
        xy = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
        heights = np.arange(
            float(nav_level) + float(min_height),
            float(nav_level) + float(max_height) + 1e-6,
            float(height_step),
        )
        query = np.concatenate(
            [
                np.column_stack((xy, np.full(len(xy), height, dtype=float)))
                for height in heights
            ],
            axis=0,
        )
        log_odds = np.asarray(
            self.map.interpolate(query, InterpolationMode.NEAREST), dtype=np.float32
        ).reshape(len(heights), self.grid_size, self.grid_size)
        occupied = np.any(log_odds > 0.6, axis=0)
        free_evidence = np.any(log_odds < -1e-5, axis=0)
        observed = self.observed_mask_2d.copy()
        known_free = observed & free_evidence & ~occupied
        unknown = ~observed
        return {
            "known_free": known_free,
            "occupied": occupied,
            "unknown": unknown,
            "observed": observed,
            "resolution": self.res,
            "origin": np.array([self.grid_min, self.grid_min], dtype=float),
        }

    def integrate_from_buffer(self):
        """
        Integrate all depth data from the buffer into the map.
        """
        for entry in self.depth_buffer:
            self.pipeline.run_pipeline(
                ["my_integrator"], wave.PosedImage(entry["pose"], entry["image"])
            )
        self.depth_buffer.clear()

        self.map.prune()  # Remove map nodes that are no longer needed

    def interpolate_occupancy_grid(self):
        """
        Get the occupancy grid from the map.
        Returns a numpy array of log odds values.
        """
        points_log_odds = self.map.interpolate(
            self.query_space, InterpolationMode.NEAREST
        )
        points_log_odds = points_log_odds.reshape(-1)
        self.occ_pts = self.query_space[points_log_odds > 0.6]
        self.free_pts = self.query_space[points_log_odds < -1e-5]

    def get_occupancy_grid(self):
        return {"occupied": self.occ_pts, "free": self.free_pts}

    def close(self):
        self.depth_buffer.clear()
        self.occ_pts = None
        self.free_pts = None
        self.query_space = None
        self.pipeline = None
        self.map = None
        self.observed_mask_2d = None
        gc.collect()
