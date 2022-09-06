import random
import time


class SimulationNewRecovery(object):
    def __init__(self, tasks, agents, task_distribution=None):
        self.tasks = tasks
        self.task_distribution = task_distribution,
        self.agents = agents
        self.time = 0
        self.agents_cost = 0
        self.start_times = []
        self.agents_pos_now = set()
        self.agents_moved = set()
        self.actual_paths = {}
        self.algo_time = 0
        self.initialize_simulation()

    def initialize_simulation(self):
        for t in self.tasks:
            self.start_times.append(t['start_time'])
        for agent in self.agents:
            self.actual_paths[agent['name']] = [{'t': 0, 'x': agent['start'][0], 'y': agent['start'][1]}]

    def time_forward(self, algorithm):
        self.time = self.time + 1
        # print('Time:', self.time)
        start_time = time.time()
        algorithm.time_forward()
        self.algo_time += time.time() - start_time
        self.agents_pos_now = set()
        self.agents_moved = set()
        agents_to_move = self.agents
        random.shuffle(agents_to_move)
        for agent in agents_to_move:
            current_agent_pos = self.actual_paths[agent['name']][-1]
            if len(algorithm.get_token()['agents'][agent['name']]) == 1:
                self.agents_moved.add(agent['name'])
                self.actual_paths[agent['name']].append(
                    {'t': self.time, 'x': current_agent_pos['x'], 'y': current_agent_pos['y']})
        # Check moving agents doesn't collide with others
        agents_to_move = [x for x in agents_to_move if x['name'] not in self.agents_moved]
        moved_this_step = -1
        while moved_this_step != 0:
            moved_this_step = 0
            for agent in agents_to_move:
                current_agent_pos = self.actual_paths[agent['name']][-1]
                if True:
                    if len(algorithm.get_token()['agents'][agent['name']]) > 1:
                        x_new = algorithm.get_token()['agents'][agent['name']][1][0]
                        y_new = algorithm.get_token()['agents'][agent['name']][1][1]
                        if True:  # tuple([x_new, y_new]) not in self.agents_pos_now or \
                            #    tuple([x_new, y_new]) == tuple(tuple([current_agent_pos['x'], current_agent_pos['y']])):
                            self.agents_moved.add(agent['name'])
                            # moved_this_step = moved_this_step + 1
                            algorithm.get_token()['agents'][agent['name']] = algorithm.get_token()['agents'][
                                                                                 agent['name']][1:]
                            self.actual_paths[agent['name']].append({'t': self.time, 'x': x_new, 'y': y_new})
                            self.agents_cost += 1
            agents_to_move = [x for x in agents_to_move if x['name'] not in self.agents_moved]
        for agent in agents_to_move:
            current_agent_pos = self.actual_paths[agent['name']][-1]
            if True:
                self.actual_paths[agent['name']].append(
                    {'t': self.time, 'x': current_agent_pos['x'], 'y': current_agent_pos['y']})
                self.agents_cost += 1

    def get_time(self):
        return self.time

    def get_algo_time(self):
        return self.algo_time

    def get_actual_paths(self):
        return self.actual_paths

    def get_new_tasks(self):
        new = []
        for t in self.tasks:
            if t['start_time'] == self.time:
                new.append(t)
        return new
