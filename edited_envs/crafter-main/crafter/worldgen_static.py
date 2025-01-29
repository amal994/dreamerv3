import numpy as np
import pathlib
import pandas as pd
from . import objects

def generate_world(world, player, mapfile, objfile):
  tunnels = np.zeros(world.area, bool) #TODO: What do tunnels doo ?
  
  materials_data = load_csv_data(mapfile)
  for x in range(world.area[0]): #TODO: This can be shortened
    for y in range(world.area[1]):
      _set_material(world, (x, y), tunnels, materials_data)

  if objfile is not None:
    entities_data = load_csv_data(objfile)
    for x in range(world.area[0]):
      for y in range(world.area[1]):
        _set_object(world, (x, y), player, entities_data)

def load_csv_data(mapfile):
  """
  Loads supplied csv file
  """
  root = pathlib.Path(__file__).parent
  dataframe = pd.read_csv(root/'staticmaps'/mapfile, header=None)
  return dataframe.values

def _set_material(world, pos, tunnels, materials_data):
  """
  Reads material symbols from materials_data and assigns them to the world locations
  W => water
  G => grass
  O => stone
  P => path
  S => sand
  T => tree
  L => lava
  """
  """
  Newly added materials
  E => coffee plant
  U => sugar cane
  R => hot sauce
  X => special spice
  N => spinach
  V => stove
  Q => grinder
  M => mason jar
  A => microwave
  K => chocolate plant
  B => beaker
  C => coffee_powder
  """
  x, y = pos
  material = materials_data[y, x].strip()
  if material == 'W': 
    world[x, y] = 'water'
  elif material == 'G':
    world[x, y] = 'grass'
  elif material == 'A':
    world[x, y] = 'microwave'
  elif material == 'K':
    world[x, y] = 'chocolate_plant'
  elif material == 'B':
    world[x, y] = 'beaker'
  elif material == 'C':
    world[x, y] = 'coffee_powder'
  elif material == 'O': 
    world[x, y] = 'stone'
  elif material == 'P':
    world[x, y] = 'path'
    tunnels[x, y] = True
  elif material == 'S':
    world[x, y] = 'sand'
  elif material == 'T':
    world[x, y] = 'tree'
  elif material == 'L':
    world[x, y] = 'lava'
  elif material == 'E':
    world[x, y] = 'coffee_plant'
  elif material == 'U':
    world[x, y] = 'sugar_cane'
  elif material == 'R':
    world[x, y] = 'hot_sauce'
  elif material == 'X':
    world[x, y] = 'spice'
  elif material == 'N':
    world[x, y] = 'spinach'
  elif material == 'V':
    world[x, y] = 'stove'
  elif material == 'Q':
    world[x, y] = 'grinder'
  elif material == 'M':
    world[x, y] = 'mason_jar'
  elif material == 'H':
    world[x, y] = 'cow'
  else:
    world[x, y] = 'sand'

def _set_object(world, pos, player, entities_data):
  pass
  """
  Reads object symbols from entities_data and assigns them to the world locations
  C => cow
  Z => zombie
  S => skeleton
  """
  x, y = pos
  dist = np.sqrt((x - player.pos[0]) ** 2 + (y - player.pos[1]) ** 2)

  entity = str(entities_data[y, x]).strip()
  if dist == 0:
    pass
  elif entity == 'C': 
    world.add(objects.Cow(world, (x, y), is_static=True))
  elif entity == 'Z':
    world.add(objects.Zombie(world, (x, y), player, is_static=True))
  elif entity == 'S': 
    world.add(objects.Skeleton(world, (x, y), player, is_static=True))
