from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Callable
from tqdm import tqdm
import numpy as np



class ProgressBar:
    def __init__(self, total_frames: int, fps: int) -> None:
        self.progress = tqdm(total=total_frames, desc=f"Rendering {total_frames / fps:.2g}s of Video", unit="frame")

    def update(self, *args) -> None:
        self.progress.update()
    
    def __enter__(self, *args) -> Callable:
        return self.update

    def __exit__(self, *args) -> None:
        self.progress.close()

class NBodySimulation:
    def __init__(self, n_bodies=3, mass=1e24, velocity_range=1e3, seed=3, seconds=10, body_size=10, trail_length=60) -> None:
        self.n_bodies = n_bodies
        self.mass = mass
        self.radius = 1e7
        self.velocity_range = velocity_range
        self.body_size = body_size
        self.trail_length = trail_length
        self.G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
        np.random.seed(seed)
        
        self.positions = np.random.uniform(-self.radius, self.radius, (n_bodies, 3))
        self.velocities = np.random.uniform(-velocity_range, velocity_range, (n_bodies, 3))
        self.velocities -= np.mean(self.velocities, axis=0)
        self.masses = np.full(n_bodies, mass)

        self.t_span = (0, round(1e4 * seconds * 60 / 200))
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], seconds * 60)
        self.fps = 60
    
    def compute_accelerations(self, positions: np.ndarray) -> np.ndarray:
        accelerations = np.zeros((self.masses.shape[0], 3))
        for i in range(self.masses.shape[0]):
            for j in range(self.masses.shape[0]):
                if i != j:
                    r = positions[j] - positions[i]
                    distance = np.linalg.norm(r) + 1e-10
                    accelerations[i] += self.G * self.masses[j] * r / distance**3
        return accelerations

    def n_body_derivatives(self, t: float, state: np.ndarray) -> np.ndarray:
        n = self.n_bodies
        positions = state[:3*n].reshape((n, 3))
        velocities = state[3*n:].reshape((n, 3))
        accelerations = self.compute_accelerations(positions)
        return np.concatenate((velocities.flatten(), accelerations.flatten()))

    def simulate(self) -> None:
        initial_state = np.concatenate((self.positions.flatten(), self.velocities.flatten()))

        solution = solve_ivp(
            self.n_body_derivatives, 
            self.t_span, 
            initial_state,
            t_eval=self.t_eval, 
            rtol=1e-6, 
            atol=1e-9
        )

        # Set frame size to 10% larger than the max min coordinates of the simulation
        self.positions_over_time = solution.y[: 3 * self.n_bodies].reshape((self.n_bodies, 3, -1))
        self.x_positions = self.positions_over_time[:, 0, :]
        self.y_positions = self.positions_over_time[:, 1, :]
        self.x_min = np.min(self.x_positions * 1.1)
        self.x_max = np.max(self.x_positions * 1.1)
        self.y_min = np.min(self.y_positions * 1.1)
        self.y_max = np.max(self.y_positions * 1.1)

    def animate(self, save_path="n_body_simulation_1080p.mp4") -> None:
        n = self.n_bodies
        fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")
        ax.axis("off")
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n))
        
        points = [ax.plot([], [], 'o', markersize=self.body_size, color=colors[i], zorder=2)[0] for i in range(n)]
        background_color = np.array([30/255, 30/255, 30/255])
        trail_colors = [(np.array(colors[i][:3]) * 0.6 + background_color * 0.4).clip(0, 1) for i in range(n)]
        trails = [
            [
                ax.plot([], [], '-', lw=self.body_size, color=trail_colors[i], alpha=1.0,
                solid_capstyle='round', zorder=1)[0] 
                for _ in range(self.trail_length)
            ]
            for i in range(n)
        ]
        
        def update(frame: int) -> list[Line2D]:
            for i, (point, trail) in enumerate(zip(points, trails)):
                point.set_data([self.positions_over_time[i, 0, frame]],
                               [self.positions_over_time[i, 1, frame]])
                start = max(0, frame - self.trail_length)
                x_trail: np.ndarray = self.positions_over_time[i, 0, start:frame]
                y_trail: np.ndarray = self.positions_over_time[i, 1, start:frame]
                for j, segment in enumerate(trail):
                    if j < x_trail.shape[0] - 1:
                        segment.set_data([x_trail[j], x_trail[j+1]], [y_trail[j], y_trail[j+1]])
                        segment.set_alpha(0.6 * ((j/len(trail)) ** 0.8))
                    else:
                        segment.set_data([], [])
            return points + [seg for t in trails for seg in t]
        
        with ProgressBar(len(self.t_eval), self.fps) as progressBar:
            ani = FuncAnimation(fig, update, frames=len(self.t_eval), interval=50, blit=True)
            ani.save(
                save_path, writer="ffmpeg", 
                fps=self.fps, dpi=120, codec="libx264", 
                extra_args=["-pix_fmt", "yuv420p", "-profile:v", "baseline"],
                progress_callback=progressBar)

if __name__ == "__main__":
    sim = NBodySimulation(n_bodies=3, mass=1e24, velocity_range=1e3, seed=3, seconds=3)
    sim.simulate()
    sim.animate()

    