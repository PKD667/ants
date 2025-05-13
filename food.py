# different types of food distribution

import random

# nouritture en tache, dispersée
# assez efficace, et simple a optimiser
def spoty(world, density=0.1):
    for i in range(world.width):
        for j in range(world.height):
            if random.random() < density:
                world.add_food(i,j)

# petits patches de nourriture (intéréssant, nécéssite des adaptations plus complexes)
def patchy(world, density=0.1, patch_size=5):
    for i in range(0,world.width,patch_size):
        for j in range(0,world.height,patch_size):
            if random.random() < density:
                for x in range(i,i+patch_size):
                    for y in range(j,j+patch_size):
                        if x < world.width and y < world.height:
                            world.add_food(x,y)

# Nouriturre sru une ligne horizontale ou verticale
def line(world):
    num_lines = world.height // 20
    if num_lines == 0: num_lines = 1 # Ensure at least one line if height is small
    for _ in range(num_lines): # Iterate for each line
        y = random.randint(0, world.height -1) # Random y for each line
        # Determine if line is horizontal or vertical randomly, or make it a parameter
        if random.random() < 0.5: # Horizontal line
            for x_coord in range(world.width):
                world.add_food(x_coord,y)
        else: # Vertical line
            x = random.randint(0, world.width -1)
            for y_coord in range(world.height):
                world.add_food(x, y_coord)

# Nourriture dans un cercle (expérimental)
def circle(world):
    for i in range(world.width):
        for j in range(world.height):
            if (i-world.width//2)**2 + (j-world.height//2)**2 < 100:
                world.add_food(i,j)

def distribute(world,config,density=0.1) :
    if config == "spoty":
        spoty(world,density)
    elif config == "patchy":
        patchy(world,density)
    elif config == "line":
        line(world)
    elif config == "circle":
        circle(world)
    else:
        print("Unknown food distribution ",config)
        return None