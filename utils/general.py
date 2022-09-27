## General helper functions

def sort_on_x(x,y,z):
    zipped = list(zip(x,y,z))
    zipped.sort(key=lambda x:x[0])
    x,y,z = list(zip(*zipped))
    return x,y,z
