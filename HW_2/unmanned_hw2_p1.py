"""

HW2 P1 

Writing function tips:
    The way to think about writing functions:
        What are the inputs?
        What are the outputs?
        What will happen in the blackbox/function?


For function to is_inside_obstacle():
    What are the inputs?
        Obstacle location [x,y]
        Obstacle size: radius
        Current location [x,y]
    What are the outputs?
        T/F
        if inside, return true
        if outside, return false
    what will happen in the blackbox?
        Compute euclidean distance from current location -> obstacle location
        store value in a variable: distance from:

        if d > r
            return False
        else d <=r
            return True

"""

import numpy as np
import matplotlib.pyplot as plt

# def is_inside_obstacle(obs_x:float, obs_y:float, obs_radius:float, curr_x:float, curr_y:float) -> bool:
#     dist_from = np.sqrt((curr_x - obs_radius)**2 + (curr_y - obs_radius)**2)
#     if dist_from > obs_radius:
#         return False
    
#     return True

min_x = 0
min_y = 0
max_x = 10
max_y = 10
obs_x = 5
obs_y = 5
obs_radius = 0.25

explorer_x = 4
explorer_y = 4

# if is_inside_obstacle(obs_x, obs_y, obs_radius, explorer_x, explorer_y):
#     print("You're dead :( ")
# else:
#     print("You're okay :) ")


# plot circle in a tuple of coordinates and radius
agent_plot = plt.Circle((explorer_x, explorer_y), 0.5, color="red")
obstacle_plot = plt.Circle((obs_x,obs_y), obs_radius, color="blue")
fig, ax = plt.subplots() # for some reason subplots and not subplot - does it matter?

ax.add_patch(obstacle_plot)
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)
plt.show()  


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
    
some_obs = Obstacle(obs_x, obs_y, obs_radius)

    # check if something is there
if some_obs.is_inside(explorer_x, explorer_y):
        print("You're dead")
else:   
        print("You're okay")    