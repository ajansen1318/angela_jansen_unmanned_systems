
import numpy as np
import matplotlib.pyplot as plt

class Obstacle:
    def __init__(self, x_pos:float, y_pos:float, radius:float) -> None:
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius

    def is_inside(self, curr_x:float, curr_y:float):
        dist_from = np.sqrt((curr_x - self.x_pos)**2 + (curr_y - self.y_pos)**2)
        if dist_from >self.radius:
            return False
        
        return True
    
if __name__ == '__main__':

    obstacle_positions = [(1,1), (4,4), (3,4), (5,0)]
    obstacle_list = [] #store obstacle classes
    obstacle_radius = 0.25

    for obs_pos in obstacle_positions:
        #print("obstacle_positions", obs_pos)
        #store obstacle info in obstacle list
        obstacle = Obstacle(obs_pos[0], obs_pos[1], obstacle_radius)
        obstacle_list.append(obstacle)

agent_x = 1
agent_y = 1.2
fig, ax = plt.subplots()
ax.set_xlim(0,10)
ax.set_ylim(0,10)

for obs in obstacle_list:
    print("This obstacle's position is", obs.x_pos, obs.y_pos)
    if obs.is_inside(agent_x, agent_y):
        print("You're dead at", obs.x_pos, obs.y_pos)
        break

for obs in obstacle_list:
    obs_plot = plt.Circle((obs.x_pos, obs.y_pos), obs.radius, color="blue")
    ax.add_patch(obs_plot)
    
plt.show()
