from world import World, Entity
 
import pygame

# rendu graphique
# joli mais inneficace

def set_sq(screen,pos,color,zoom=1) :
    for i in range(zoom):
        for j in range(zoom):
            screen.set_at((pos[0]*zoom+i,pos[1]*zoom+j),color)

def render(screen,world,zoom=1):

    screen.fill((0, 0, 0))

    for entity in world.entities:
        #print(entity.x, entity.y)
        set_sq(screen,(entity.x,entity.y),(255,255,255),zoom)

    # draw the food grid in green
    for x in range(world.width):
        for y in range(world.height):
            if world.get_food(x, y) > 0:
                # set pixel color to green
                set_sq(screen,(x,y),(0,255,0),zoom)
