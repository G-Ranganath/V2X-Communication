import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations
import csv

# --- 1. Core Logic Classes ---
class BeamCodebook:
    def __init__(self, num_antennas=16, num_beams=32):
        self.N, self.M = num_antennas, num_beams
        self.angles = np.linspace(-np.pi, np.pi, num_beams) 
        self.codebook = np.array([np.exp(1j * np.pi * np.arange(self.N) * np.sin(a)) / np.sqrt(self.N) for a in self.angles])

    def get_best_beam(self, target_angle_rad):
        v_vec = np.exp(1j * np.pi * np.arange(self.N) * np.sin(target_angle_rad)) / np.sqrt(self.N)
        gains = np.abs(np.conj(self.codebook) @ v_vec)**2
        return np.argmax(gains)

class RSU:
    def __init__(self, rsu_id, x, y):
        self.rsu_id, self.x, self.y = rsu_id, x, y
        self.codebook = BeamCodebook()

class Vehicle:
    def __init__(self, veh_id, start_x, start_y, target_speed_kmh, direction):
        self.veh_id, self.x, self.y = veh_id, start_x, start_y
        self.direction = direction 
        self.target_speed = target_speed_kmh * (1000 / 3600)
        self.current_speed = self.target_speed
        self.status = "GO"
        self.passed_intersection = False
        self.deceleration_rate = 8.0 # m/s^2 for braking

    def move(self, dt):
        if self.direction == 'E':   self.x += self.current_speed * dt
        elif self.direction == 'W': self.x -= self.current_speed * dt
        elif self.direction == 'N': self.y += self.current_speed * dt
        elif self.direction == 'S': self.y -= self.current_speed * dt

    def brake(self):
        self.current_speed = max(0, self.current_speed - self.deceleration_rate) 
        self.status = "STOP"

    def accelerate(self):
        self.current_speed = min(self.target_speed, self.current_speed + 3.0)
        self.status = "GO"

# --- 2. Setup 4-Way Intersection Environment ---
rsus = [RSU(1, -50, 50), RSU(2, 50, 50), RSU(3, -50, -50), RSU(4, 50, -50)]

vehicles = [
    Vehicle(1, -300, -4, 90, 'E'), 
    Vehicle(2, 350, 4, 110, 'W'),  
    Vehicle(3, 4, -250, 80, 'N'),   
    Vehicle(4, -4, 400, 100, 'S'),  
    Vehicle(5, -450, -4, 120, 'E'), 
    Vehicle(6, 4, -400, 95, 'N')    
]

# --- 3. CSV Data Logging Setup ---
csv_file = open('intersection_training_data.csv', 'w', newline='')
writer = csv.writer(csv_file)
writer.writerow(['timestamp', 'veh_id', 'x', 'y', 'speed', 'dist_to_center', 'dist_to_closest_veh', 'time_to_brake', 'best_beam', 'action_label'])

# --- 4. Animation Setup ---
fig, ax = plt.subplots(figsize=(12, 12)) 
ax.set_xlim(-250, 250); ax.set_ylim(-250, 250)
ax.set_facecolor('#1e1e1e') # Dark mode for better HUD visibility
ax.set_title("V2X Smart Intersection: Predictive Braking HUD", color='white', fontsize=14)

ax.fill_between([-250, 250], -15, 15, color='#333333') # Horizontal Road
ax.fill_betweenx([-250, 250], -15, 15, color='#333333') # Vertical Road

for r in rsus:
    ax.scatter(r.x, r.y, c='cyan', marker='^', s=200, zorder=10)
    ax.text(r.x, r.y+10, f"RSU {r.rsu_id}", ha='center', color='cyan', fontweight='bold')

veh_plots = [ax.scatter([], [], c='lime', marker='s', s=100, zorder=15) for _ in vehicles]
# The Heads-Up Display (HUD) text for each vehicle
veh_huds = [ax.text(0, 0, '', ha='left', fontsize=8, color='white', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')) for _ in vehicles]

v2v_pairs = list(combinations(range(6), 2))
v2v_lines = [ax.plot([], [], color='magenta', linestyle=':', linewidth=1)[0] for _ in v2v_pairs]

intersection_manager = {'occupied_by': None}

def update(frame):
    dt = 0.1 
    
    for v in vehicles:
        dist_to_center = np.hypot(v.x, v.y)
        
        # --- PHYSICS CALCULATIONS FOR HUD ---
        # 1. How much distance is needed to stop completely? (v^2 / 2a)
        braking_dist_needed = (v.current_speed ** 2) / (2 * v.deceleration_rate)
        # 2. How much time until the driver MUST hit the brakes?
        safety_buffer = 20.0 # Stop 20m before the exact center
        distance_until_brake = dist_to_center - braking_dist_needed - safety_buffer
        
        time_to_brake = distance_until_brake / v.current_speed if v.current_speed > 0 else 0
        
        if time_to_brake < 0 and not v.passed_intersection:
            brake_warning = "CRITICAL: BRAKING!"
            hud_color = 'red'
        elif time_to_brake < 3.0 and not v.passed_intersection:
            brake_warning = f"Brake in: {time_to_brake:.1f}s"
            hud_color = 'yellow'
        else:
            brake_warning = "Safe"
            hud_color = 'lime'

        # --- INTERSECTION LOGIC ---
        if (v.direction == 'E' and v.x > 15) or (v.direction == 'W' and v.x < -15) or \
           (v.direction == 'N' and v.y > 15) or (v.direction == 'S' and v.y < -15):
            v.passed_intersection = True
            v.accelerate()
            if intersection_manager['occupied_by'] == v.veh_id:
                intersection_manager['occupied_by'] = None 
                
        elif dist_to_center < 100 and not v.passed_intersection:
            if intersection_manager['occupied_by'] is None or intersection_manager['occupied_by'] == v.veh_id:
                intersection_manager['occupied_by'] = v.veh_id
                v.accelerate()
            else:
                v.brake()
        else:
            v.accelerate() 

        v.move(dt)
        veh_plots[vehicles.index(v)].set_offsets([[v.x, v.y]])
        
        # --- UPDATE HUD DISPLAY ---
        speed_kmh = int(v.current_speed * 3.6)
        hud_text = f"ID: V{v.veh_id}\nSpd: {speed_kmh} km/h\nDist: {int(dist_to_center)}m\n{brake_warning}"
        veh_huds[vehicles.index(v)].set_position((v.x + 10, v.y + 10))
        veh_huds[vehicles.index(v)].set_text(hud_text)
        veh_huds[vehicles.index(v)].set_color(hud_color)

        # --- DATA LOGGING ---
        dist_to_closest = 999.0
        for other_v in vehicles:
            if other_v.veh_id != v.veh_id:
                d = np.hypot(v.x - other_v.x, v.y - other_v.y)
                if d < dist_to_closest: dist_to_closest = d

        closest_rsu = min(rsus, key=lambda r: np.hypot(r.x - v.x, r.y - v.y))
        best_beam_idx = closest_rsu.codebook.get_best_beam(np.arctan2(v.y - closest_rsu.y, v.x - closest_rsu.x))
        action_label = 1 if v.status == "GO" else 0

        writer.writerow([frame * dt, v.veh_id, v.x, v.y, v.current_speed, dist_to_center, dist_to_closest, max(0, time_to_brake), best_beam_idx, action_label])

    # --- DRAW V2V DISTANCE LINES ---
    for idx, (i, j) in enumerate(v2v_pairs):
        v1, v2 = vehicles[i], vehicles[j]
        dist = np.hypot(v1.x - v2.x, v1.y - v2.y)
        if dist < 120: # Only show connection if they are close
            v2v_lines[idx].set_data([v1.x, v2.x], [v1.y, v2.y])
        else:
            v2v_lines[idx].set_data([], [])

    return veh_plots + veh_huds + v2v_lines

ani = FuncAnimation(fig, update, frames=800, interval=30, blit=True, repeat=False)
plt.show()
csv_file.close()
