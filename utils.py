import random
def position_distribution(type,WIDTH,HEIGHT):
    if type == "random":
        return random.randint(0,WIDTH-1),random.randint(0,HEIGHT-1)
    elif type == "center":
        return WIDTH//2,HEIGHT//2
    elif type == "top_left":
        return 0,0
    elif type == "top_right":
        return WIDTH-1,0
    elif type == "bottom_left":
        return 0,HEIGHT-1
    elif type == "bottom_right":
        return WIDTH-1,HEIGHT-1
    else:
        return None,None
