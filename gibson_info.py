import json
import numpy as np
from habitat_test_env import HabitatTestEnv
import os
import re
from collections import Counter

class_labels = sorted(['bed', 'chair', 'couch', 'dining table', 'toilet'])

level_override = {
    'Airport': 3,
    'Albertville': 3,
    'Allensville': 1,
    'Anaheim': 3,
    'Ancor': 3,
    'Andover': 2,
    'Annona': 2,
    'Adairsville': 3,
    'Arkansaw': 3,
    'Athens': None,
    'Bautista': 3,
    'Beechwood': 2,
    'Benevolence': 3,
    'Bohemia': 2,
    'Bonesteel': 2,
    'Bonnie': 2,
    'Broseley': None,
    'Brown': None,
    'Browntown': 2,
    'Byers': 3,
    'Castor': 2,
    'Chilhowie': None,
    'Churchton': 3,
    'Clairton': None,
    'Coffeen': 4,
    'Collierville': 3,
    'Corozal': 2,
    'Cosmos': 2,
    'Cottonport': None,
    'Darden': 3,
    'Duarte': None,
    'Eagan': 3,
    'Emmaus': None,
    'Forkland': 3,
    'Frankfort': None,
    'Globe': None,
    'Goffs': 2,
    'Goodfield': 3,
    'Goodwine': 2,
    'Goodyear': 3,
    'Gravelly': None,
    'Hainesburg': 2,
    'Hanson': 3,
    'Highspire': None,
    'Hildebran': 1,
    'Hillsdale': 2,
    'Hiteman': 3,
    'Hominy': 2,
    'Hordville': None,
    'Hortense': 3,
    'Irvine': None,
    'Kemblesville': 1,
    'Klickitat': 3,
    'Kobuk': 2,
    'Lakeville': 2,
    'Leonardo': 3,
    'Lindenwood': 2,
    'Lynchburg': 2,
    'Maida': 1,
    'Markleeville': 2,
    'Marland': 3,
    'Marstons': 4,
    'Martinville': None,
    'Maugansville': None,
    'Merom': 2,
    'Micanopy': 3,
    'Mifflinburg': 3,
    'Musicks': 2,
    'Neibert': 2,
    'Neshkoro': None,
    'Newcomb': None,
    'Newfields': 3,
    'Nuevo': 2,
    'Onaga': 2,
    'Oyens': 1,
    'Pablo': 1,
    'Pamelia': 2,
    'Parole': 1,
    'Pearce': None,
    'Pinesdale': 2,
    'Pittsburg': None,
    'Pomaria': 3,
    'Potterville': None,
    'Ranchester': 2,
    'Readsboro': None,
    'Rogue': 3,
    'Rosser': 1,
    'Sands': 2,
    'Scioto': 4,
    'Shelbiana': 1,
    'Shelbyville': 3,
    'Silas': 2,
    'Soldier': 2,
    'Southfield': 2,
    'Springerville': None,
    'Stilwell': 2,
    'Stockman': 3,
    'Sugarville': None,
    'Sunshine': None,
    'Sussex': 2,
    'Sweatman': None,
    'Swisshome': 3,
    'Swormville': 1,
    'Thrall': None,
    'Tilghmanton': 2,
    'Timberon': None,
    'Tokeland': 2,
    'Tolstoy': 2,
    'Touhy': 2,
    'Tyler': None,
    'Victorville': None,
    'Wainscott': 2,
    'Waipahu': None,
    'Westfield': None,
    'Wiconisco': 3,
    'Willow': None,
    'Wilseyville': 3,
    'Winooski': None,
    'Woodbine': 2,
    'Wyldwood': 3
}

levels_from_env = {
    'Athens': 4,
    'Broseley': 3,
    'Brown': 4,
    'Chilhowie': 2,
    'Clairton': 1,
    'Cottonport': 3,
    'Duarte': 3,
    'Emmaus': 3,
    'Frankfort': 2,
    'Globe': 3,
    'Gravelly': 2,
    'Highspire': 3,
    'Hordville': 3,
    'Irvine': 3,
    'Martinville': 2,
    'Maugansville': 2,
    'Neshkoro': 2,
    'Newcomb': 1,
    'Pearce': 2,
    'Pittsburg': 3,
    'Potterville': 4,
    'Readsboro': 3,
    'Springerville': 3,
    'Sugarville': 2,
    'Sunshine': 6,
    'Sweatman': 1,
    'Thrall': 2,
    'Timberon': 4,
    'Tyler': 2,
    'Victorville': 3,
    'Waipahu': 3,
    'Westfield': 3,
    'Willow': 2,
    'Winooski': 5
}

colors = {
    'bed': (175, 124, 222),
    'chair': (64, 207, 255),
    'couch': (195, 255, 54),
    'dining table': (245, 66, 66),
    'toilet': (227, 159, 82)
}

# missing = [k for k,v in level_override.items() if v is None]


class GibsonHouse:
    def __init__(self, dataobj):
        self.name = dataobj['id']
        self._semantics = None
        self.data = dataobj

    @property
    def semantics(self):
        if self._semantics is None:
            if self.data['split_tiny'] != 'none':
                folder = '/scratch/mc48/gibson_tiny_annotations/verified_graph'
            else:
                folder = '/scratch/mc48/3DSceneGraph_medium'
            self._semantics = np.load(f'{folder}/3DSceneGraph_{self.name}.npz',
                                      allow_pickle=True)['output'][()]
        return self._semantics

    @property
    def toilets(self):
        toilets = [
            o for i, o in self.semantics['object'].items()
            if o['class_'] == 'toilet'
        ]
        return toilets

    @property
    def num_floors(self):
        if 'num_floors' in self.semantics['building'].keys():
            print('sem')
            return self.semantics['building']['num_floors']
        elif level_override[self.name]:
            print('online')
            return level_override[self.name]
        else:
            print('gib')
            return min(self.data['stats']['floor'], levels_from_env[self.name])

    @property
    def object_locations(self):
        locations = {
            c: [
                o['location'] for i, o in self.semantics['object'].items()
                if o['class_'] == c
            ]
            for c in class_labels
        }
        return {
            k: list(map(self.gibson_to_habitat_coordinates, locs))
            for k, locs in locations.items()
        }

    @property
    def objects(self):
        out = {}
        for cls in class_labels:
            points = []
            objs = [
                o for o in self.semantics['object'].values()
                if o['class_'] == cls
            ]
            objs = map(
                lambda x: (self.gibson_to_habitat_coordinates(x['location']),
                           self.gibson_to_habitat_coordinates(x['size'])),
                objs)
            for loc, size in objs:

                def to_pts(arg):
                    x, y = arg
                    return np.array(
                        (loc[0] + x * size[0], loc[1], loc[2] + y * size[2]))

                points.append(
                    list(
                        map(to_pts, [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5),
                                     (-0.5, 0.5)])))
            out[cls] = points
        return out

    @property
    def object_locations_for_habitat_dest(self):
        out = {}
        for cls in class_labels:
            points = []
            objs = [
                o for o in self.semantics['object'].values()
                if o['class_'] == cls
            ]
            objs = map(
                lambda x: (self.gibson_to_habitat_coordinates(x['location']),
                           self.gibson_to_habitat_coordinates(x['size'])),
                objs)
            for loc, size in objs:
                for x, y in [(0.5, 0.5), (0.5, -0.5), (-0.5, -0.5),
                             (-0.5, 0.5)]:
                    points.append(
                        np.array((loc[0] + x * size[0], loc[1],
                                  loc[2] + y * size[2])))
            out[cls] = points
        return out

    @property
    def toilet_locations_habitat(self):
        return [
            self.gibson_to_habitat_coordinates(t['location'])
            for t in self.toilets
        ]

    # based on https://github.com/facebookresearch/habitat-sim/blob/f43e096abee394432e4a1fd1323bd8b8fc2448e6/src/utils/datatool/datatool.cpp
    # the transformationfrom gibson to habitat is to rotate the Y axis to the Z axis. Which could be done multiple ways, but it looks like
    # theirs rotates around the x axis
    def gibson_to_habitat_coordinates(self, point):
        return np.array([point[0], point[2], -point[1]])

    def get_env(self, **kwargs):
        return HabitatTestEnv(f'/scratch/mc48/gibson/{self.name}.glb',
                              **kwargs)


def get_houses(split=['train', 'val']):
    with open('/scratch/mc48/gibson/metadata.json') as json_file:
        data = json.load(json_file)
    houses = [d for d in data if d['split_tiny'] in split]
    return [GibsonHouse(d) for d in houses]


def get_houses_medium(split=['train', 'val']):
    with open('/scratch/mc48/gibson/metadata.json') as json_file:
        data = json.load(json_file)
    houses = [d for d in data if d['split_medium'] in split]
    return [
        GibsonHouse(d['id'], d['split_medium'], 'medium', d) for d in houses
    ]


def get_house(name):
    with open('/scratch/mc48/gibson/metadata.json') as json_file:
        data = json.load(json_file)
    return [GibsonHouse(d) for d in data if d['id'] == name][0]


def relevant_locations(agent_pos, locs):
    def lam(t):
        dist = t[1] - agent_pos[1]
        return dist < 1 and dist >= 0

    return list(filter(lam, locs))

def relevant_objects(agent_pos, objects):
    def lam(t):
        dist = t[0][1] - agent_pos[1]
        return dist < 1 and dist >= 0

    return list(filter(lam, objects))


def get_floor_count_online(name):
    html = os.popen(
        f"curl https://3dscenegraph.stanford.edu/{name}.html").read()
    match = re.search("Floors: (\d)", html)
    if match:
        return int(match[1])
    else:
        return None


def get_floor_count_env(house):
    env = house.get_env(random_goal=True)
    points = np.array(
        [env.env.sim.sample_navigable_point() for _ in range(10000)])
    counts = Counter(points[:, 1]).most_common(10)
    levels = list(filter(lambda x: x[1] > 500, counts))
    return len(levels)


# splits = {
# "tiny_train"
# }

# randomly sampled 15 houses from medium/tiny (medium without tiny)
medium_inverse_train_names = [
    'Maugansville', 'Sussex', 'Andover', 'Annona', 'Goodfield', 'Kemblesville',
    'Goodwine', 'Adairsville', 'Nuevo', 'Stilwell', 'Eagan', 'Touhy',
    'Springerville', 'Brown', 'Castor'
]


def get_house_split(split):
    with open('/scratch/mc48/gibson/metadata.json') as json_file:
        data = json.load(json_file)

    if split == 'medium_inverse_train':
        houses = [
            GibsonHouse(d) for d in data
            if d['id'] in medium_inverse_train_names
        ]
        if len(houses) != 15:
            raise Exception('bad length')
        return houses
    elif split == 'medium_train':
        houses = [GibsonHouse(d) for d in data if d['split_medium'] == 'train']
        if len(houses) != 100:
            raise Exception('bad length')
        return houses
    elif split == 'medium_qlearning_train':
        houses = [
            GibsonHouse(d) for d in data
            if d['id'] not in medium_inverse_train_names
            and d['split_medium'] == 'train'
        ]
        if len(houses) != 85:
            raise Exception('bad length')
        return houses
    elif split == 'medium_val':
        return [GibsonHouse(d) for d in data if d['split_medium'] == 'val']
    raise Exception('bad split')


if __name__ == '__main__':
    print(len(get_house_split('medium_inverse_train')))
    print(len(get_house_split('medium_qlearning_train')))
    print(len(get_house_split('medium_val')))

    # houses = get_house_split('medium_qlearning_train')
    # for h in houses:
    # print(h.name)

    # import pdb; pdb.set_trace()
    # !len(list(filter(lambda x: x is None,level_override.values())))
    # len(level_override.values())
    # houses = get_houses_medium()
    # for h in houses:
    # print(h.name, h.num_floors)
    # exit()

    # houses = get_houses_medium()
    # counts = {}
    # for h in houses:
    # print(h.name)
    # counts[h.name] = get_floor_count(h.name)
    # print(counts)
    # import pdb; pdb.set_trace()
    # counts

# if __name__ == '__main__':
# houses = get_houses_medium()
# mhouses = [h for h in houses if h.name in missing]
# env_floors = {}
# for h in mhouses:
# env_floors[h.name] = get_floor_count_env(h)
# print(env_floors)
# import pdb; pdb.set_trace()
