# Définition d'objects utilisés pour la simulation
# Plutot simple

import numpy as np
import math
import random
import torch



from brain import AntBrain


class World:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        # nourriture
        self.foodgrid = np.zeros((width, height), dtype=int)
        # pas utilisé pour l'instant
        self.chemgrid = np.zeros((width, height), dtype=int)
        self.food = 0
        self.entities = []

    def reset(self):
        self.foodgrid = np.zeros((self.width, self.height), dtype=int)
        self.chemgrid = np.zeros((self.width, self.height), dtype=int)
        self.food = 0
        self.entities = []

    def copy(self):
        new_world = World(self.width,self.height)
        new_world.foodgrid = np.copy(self.foodgrid)
        new_world.chemgrid = np.copy(self.chemgrid)
        new_world.food = self.food
        return new_world

    def add_food(self, x, y):
        self.foodgrid[x][y] = 1
        self.food += 1

    def remove_food(self, x, y):
        self.foodgrid[x][y] = 0
        self.food -= 1
    
    def get_food(self, x, y):
        try :
            return self.foodgrid[x][y]
        except IndexError:
            return 0
    
    def get_food_count(self):
        return self.food
    
    def add_chemical(self, x, y):
        self.chemgrid[x][y] += 1
    
    def remove_chemical(self, x, y):
        self.chemgrid[x][y] -= 1
    
    def get_chemical(self, x, y):
        try :
            return self.chemgrid[x][y]
        except IndexError:
            return 0
    
    def get_ditance(self,en1,en2):
        x1,y1 = en1.get_position()
        x2,y2 = en2.get_position()
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
        

    # add an entity to the world
    def add_entity(self, entity):
        self.entities.append(entity)

class Entity:

    def __init__(self, x, y, size=1):
        self.x = x
        self.y = y

        self.size = size

    def move(self,world, dx, dy):
        self.x += dx
        self.y += dy

        if self.x <= 0:
            self.x = 1
        if self.x >= world.width -1:
            self.x = world.width - 2
        if self.y <= 0:
            self.y = 1
        if self.y >= world.height-1:
            self.y = world.height - 2

    def get_position(self):
        return (self.x, self.y)

    def set_position(self, x, y):
        self.x = x
        self.y = y

    





class Ant(Entity): 
    
        def __init__(self, x, y, size=1):
            super().__init__( x, y, size)
            self.food = 0
            self.brain = AntBrain.random()

        def copy(self):
            ant = Ant(self.x, self.y)
            ant.brain = self.brain
            return ant
        
        def mutate(self):
            self.brain.mutate()
    
        def get_food(self):
            return self.food
    
        def pick_up_food(self,world):
            if world.get_food(self.x, self.y) > 0:
                world.remove_food(self.x, self.y)
                self.food += 1
                return True
            else :
                return False
    
        def drop_food(self,world):
            if self.food > 0:
                world.add_food(self.x, self.y)
                self.food -= 1
                return True
            else :
                return False
        def put_pheromone(self,world):

            c = world.get_chemical(self.x, self.y)

            world.add_chemical(self.x, self.y)
        
        def step(self,world,logger=None):
            # get the 4x4 food grid around the ant
            foodgrid = np.zeros((3, 3), dtype=int)
            for i in range(3):
                for j in range(3):
                    # if (self.x + i - 2, self.y + j - 2) is out of bounds, set foodgrid[i][j] to -1 
                    if self.x + i - 2 < 0 or self.x + i - 2 >= world.width or self.y + j - 2 < 0 or self.y + j - 2 >= world.height:
                        foodgrid[i][j] = -1
                    foodgrid[i][j] = world.get_food(self.x + i - 2, self.y + j - 2)
            
            # get the position of the ant in the world (relative to the center of the grid)
            x = self.x - world.width // 2
            y = self.y - world.height // 2
            #print(x,y)
            # create the input vector
            input = foodgrid.flatten()
            # if there are borders, make them negative values on the input vector
            


            direction = self.brain.act(input)

            # move the ant
            # 0: up, 1: right, 2: down, 3: left

            if direction == 0:
                self.move(world,0,-1)
            elif direction == 1:
                self.move(world,1,0)
            elif direction == 2:
                self.move(world,0,1)
            elif direction == 3:
                self.move(world,-1,0)
            # pick up food if there is any
            self.pick_up_food(world)  

            if logger is not None:
                logger.append({
                    "x": int(self.x),
                    "y": int(self.y),
                    "food": self.food,
                    # foodgrid converted to a 2D list
                    "foodgrid": foodgrid.tolist(),
                    "direction": direction
                })


