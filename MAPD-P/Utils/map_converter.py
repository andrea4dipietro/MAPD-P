import json
import os
import random
import re

import yaml

import RoothPath

if __name__ == '__main__':
    yaml_dic = {}
    with open(os.path.join(os.path.join(RoothPath.get_root(), 'Benchmarks'), 'den308d.map')) as ascii_map:
        ascii_map.readline()
        h = int(re.findall(r'\d+', ascii_map.readline())[0])
        w = int(re.findall(r'\d+', ascii_map.readline())[0])
        yaml_dic['agents'] = [{'start': [48, 10], 'name': 'agent0'}]
        yaml_dic['map'] = {'dimensions': [w, h], 'obstacles': [], 'non_task_endpoints': [[48, 10]],
                           'start_locations': [[50, 10]], 'goal_locations': [[54, 10]]}
        ascii_map.readline()
        free_locations = []
        start_locations = []
        goal_locations = []
        agents_locations = []
        for i in range(h - 1, -1, -1):
            line = ascii_map.readline()
            print(line)
            for j in range(w):
                if line[j] == '@' or line[j] == 'T':
                    yaml_dic['map']['obstacles'].append((j, i))
                else:
                    free_locations.append([j, i])
        start_locations = random.sample(free_locations, 15)
        free_locations = list(set(free_locations).difference(set(start_locations)))
        goal_locations = random.sample(free_locations, 10)
        free_locations = list(set(free_locations).difference(set(goal_locations)))
        agents_locations = random.sample(free_locations, 10)
        agents = []
        locs = []
        i = 0
        for loc in agents_locations:
            string = "agent" + str(i)
            agents.append({'start': [loc[0], loc[1]], 'name': string})
            locs.append((loc[0], loc[1]))
            i = i + 1
        yaml_dic['agents'] = agents
        yaml_dic['map'] = {'dimensions': [w, h], 'obstacles': yaml_dic['map']['obstacles'], 'non_task_endpoints': locs,
                           'start_locations': start_locations, 'goal_locations': goal_locations}

    with open(os.path.join(RoothPath.get_root(), 'config.json'), 'r') as json_file:
        config = json.load(json_file)
    with open(os.path.join(os.path.join(RoothPath.get_root(), config['input_path']), 'den308d.yaml'),
              'w') as param_file:
        yaml.dump(yaml_dic, param_file)
