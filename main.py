import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import csv
import os
from matplotlib.backend_bases import KeyEvent

# --- CONFIG ---
WORLD_SIZE = np.array([5.0, 5.0, 10.0])
GRAVITY_BASE = 9.81
TIMESTEP = 0.01
NUM_PARTICLES = 300
LIGHTNING_PROB = 0.01
BOND_DISTANCE = 0.2
EXPORT_PATH = "simulation_export.csv"
BOND_BREAK_PROB = 0.001

PARTICLE_TYPES = {
    "H": {"mass": 1, "valency": 1, "energy": 5, "color": "white"},
    "He": {"mass": 4, "valency": 2, "energy": 3, "color": "cyan"},
    "O": {"mass": 16, "valency": 8, "energy": 8, "color": "red"},
    "N": {"mass": 14, "valency": 5, "energy": 7, "color": "blue"},
    "C": {"mass": 12, "valency": 4, "energy": 6, "color": "black"},
    "Si": {"mass": 28, "valency": 10, "energy": 9, "color": "green"},
    "Fe": {"mass": 56, "valency": 6, "energy": 7, "color": "brown"},
    "Na": {"mass": 23, "valency": 12, "energy": 6, "color": "purple"},
}

class Particle:
    def __init__(self, element):
        self.element = element
        props = PARTICLE_TYPES[element]
        self.mass = props["mass"]
        self.valency = props["valency"]
        self.energy = props["energy"]
        self.color = props["color"]
        self.position = np.random.rand(3) * WORLD_SIZE
        self.velocity = np.random.randn(3) * (1 / self.mass)
        self.bonds = []

    def apply_gravity(self):
        height_ratio = self.position[2] / WORLD_SIZE[2]
        gravity = GRAVITY_BASE * (1 - height_ratio)
        self.velocity[2] -= gravity * TIMESTEP

    def move(self):
        self.position += self.velocity * TIMESTEP
        self.handle_boundaries()

    def handle_boundaries(self):
        for i in range(3):
            if self.position[i] < 0 or self.position[i] > WORLD_SIZE[i]:
                self.velocity[i] *= -1
                self.position[i] = np.clip(self.position[i], 0, WORLD_SIZE[i])

class Molecule:
    def __init__(self, particles):
        self.particles = particles
        self.update_properties()

    def update_properties(self):
        total_mass = sum(p.mass for p in self.particles)
        self.center_of_mass = sum(p.mass * p.position for p in self.particles) / total_mass
        self.velocity = sum(p.mass * p.velocity for p in self.particles) / total_mass

    def apply_gravity(self):
        z = self.center_of_mass[2]
        height_ratio = z / WORLD_SIZE[2]
        gravity = GRAVITY_BASE * (1 - height_ratio)
        self.velocity[2] -= gravity * TIMESTEP

    def move(self):
        for p in self.particles:
            p.position += self.velocity * TIMESTEP
            p.handle_boundaries()

    def break_bonds(self):
        for p in self.particles:
            for bonded in p.bonds[:]:
                if random.random() < BOND_BREAK_PROB:
                    p.bonds.remove(bonded)
                    bonded.bonds.remove(p)

    def update(self):
        self.update_properties()
        self.apply_gravity()
        self.move()
        self.break_bonds()

def identify_molecules(particles):
    visited = set()
    molecules = []

    def dfs(p, group):
        visited.add(p)
        group.append(p)
        for bonded in p.bonds:
            if bonded not in visited:
                dfs(bonded, group)

    for p in particles:
        if p not in visited:
            group = []
            dfs(p, group)
            molecules.append(Molecule(group))

    return molecules

def emit_light(particles, sun_pos):
    for p in particles:
        dist = np.linalg.norm(p.position - sun_pos)
        p.energy += 0.05 / (dist + 0.1)

def lightning_strike(particles):
    x, y = random.uniform(0, WORLD_SIZE[0]), random.uniform(0, WORLD_SIZE[1])
    strike_zone = []
    for p in particles:
        if abs(p.position[0] - x) < 0.5 and abs(p.position[1] - y) < 0.5:
            p.energy += 20
            strike_zone.append((x, y))
    return strike_zone

def try_bond(p1, p2):
    if len(p1.bonds) >= p1.valency or len(p2.bonds) >= p2.valency:
        return
    distance = np.linalg.norm(p1.position - p2.position)
    if distance < BOND_DISTANCE and (p1.energy + p2.energy) > 10:
        if p2 not in p1.bonds:
            p1.bonds.append(p2)
            p2.bonds.append(p1)
            p1.energy -= random.randint(1, 3)
            p2.energy -= random.randint(1, 5)

def export_state(step, particles):
    file_exists = os.path.exists(EXPORT_PATH)
    with open(EXPORT_PATH, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["step", "element", "x", "y", "z", "energy"])
        for p in particles:
            writer.writerow([step, p.element, *p.position, p.energy])

# --- Initialization ---
particles = [Particle(random.choice(list(PARTICLE_TYPES.keys()))) for _ in range(NUM_PARTICLES)]
sun_pos = np.array([WORLD_SIZE[0]/2, WORLD_SIZE[1]/2, WORLD_SIZE[2]-0.1])
export_flag = False  # Controlled by keypress

# --- Visualization ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()

def update_plot(lightning_coords):
    ax.clear()
    ax.set_xlim([0, WORLD_SIZE[0]])
    ax.set_ylim([0, WORLD_SIZE[1]])
    ax.set_zlim([0, WORLD_SIZE[2]])
    ax.set_title("Microscopic Universe Simulation")

    for p in particles:
        ax.scatter(*p.position, color=p.color, s=10)

    for p in particles:
        for bonded in p.bonds:
            if id(p) < id(bonded):
                xs = [p.position[0], bonded.position[0]]
                ys = [p.position[1], bonded.position[1]]
                zs = [p.position[2], bonded.position[2]]
                ax.plot(xs, ys, zs, color='gray', linewidth=0.5)

    for x, y in lightning_coords:
        ax.plot([x, x], [y, y], [0, WORLD_SIZE[2]], color='blue', linewidth=1)

    ax.scatter(*sun_pos, color='yellow', s=150, marker='o', alpha=0.6)
    plt.pause(0.001)

def on_key(event: KeyEvent):
    global export_flag
    if event.key == 'ctrl+e':
        export_flag = True
        print("Export triggered on step ", TIMESTEP)

fig.canvas.mpl_connect('key_press_event', on_key)

# --- Simulation loop ---
for step in range(10000):
    emit_light(particles, sun_pos)
    lightning_coords = lightning_strike(particles) if random.random() < LIGHTNING_PROB else []

    molecules = identify_molecules(particles)
    for mol in molecules:
        mol.update()

    # Spatial partitioning: skip if too far
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            if np.all(np.abs(particles[i].position - particles[j].position) < BOND_DISTANCE * 1.5):
                try_bond(particles[i], particles[j])

    if export_flag:
        export_state(step, particles)
        export_flag = False

    if step % 5 == 0:
        update_plot(lightning_coords)

plt.ioff()
plt.show()
