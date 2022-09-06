import math
import random
from collections import defaultdict
from math import fabs

import numpy

from Simulation.CBS.cbs import CBS, Environment


class TokenPassingRecovery(object):
    def __init__(self, agents, dimesions, obstacles, non_task_endpoints, simulation, starts, a_star_max_iter=800000000,
                 path_1_modified=False, path_2_modified=False, preemption_radius=0, preemption_duration=0):
        self.agents = agents
        self.starts = starts
        self.dimensions = dimesions
        self.path_1_modified = path_1_modified
        self.path_2_modified = path_2_modified
        self.preemption_radius = preemption_radius
        self.preemption_duration = preemption_duration
        preemption_zones = {}
        self.preempted_locations = {}
        self.preemption_status = {}
        for location in self.starts:
            zone = []
            for start in self.starts:
                if self.admissible_heuristic(start, location) <= preemption_radius:
                    zone.append(start)
            preemption_zones[tuple(location)] = zone
        self.preemption_zones = preemption_zones
        for agent_name in self.agents:
            self.preempted_locations[agent_name['name']] = []
        #   self.preemption_status[agent_name['name']] = 0
        self.task_distribution = numpy.array(simulation.task_distribution)

        self.obstacles = set(obstacles)
        self.non_task_endpoints = non_task_endpoints
        if len(agents) > len(non_task_endpoints):
            # #self.print('There are more agents than non task endpoints, instance is not well-formed.')
            exit(1)
        # TODO: Check all properties for well-formedness
        self.token = {}
        self.simulation = simulation
        self.a_star_max_iter = a_star_max_iter
        self.init_token()

    def init_token(self):
        self.token['agents'] = {}
        self.token['tasks'] = {}
        self.token['start_tasks_times'] = {}
        self.token['completed_tasks_times'] = {}
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        self.token['agents_to_tasks'] = {}
        self.token['completed_tasks'] = 0
        self.token['n_replans'] = 0
        self.token['path_ends'] = set()
        self.token['occupied_non_task_endpoints'] = set()
        self.token['delayed_agents'] = []
        self.token['delayed_agents_to_reach_task_start'] = []
        self.token['agent_at_end_path'] = []
        self.token['agent_at_end_path_pos'] = []
        self.token['agents_in_recovery_trial'] = []
        for a in self.agents:
            self.token['agents'][a['name']] = [a['start']]
            self.token['path_ends'].add(tuple(a['start']))
        self.token['prob_exceeded'] = False
        self.token['deadlock_count_per_agent'] = defaultdict(lambda: 0)

    def get_idle_agents(self):
        agents = {}
        for name, path in self.token['agents'].items():
            if len(path) == 1:
                agents[name] = path
        return agents

    def admissible_heuristic(self, task_pos, agent_pos):
        if task_pos is None:
            return math.inf
        return fabs(task_pos[0] - agent_pos[0]) + fabs(task_pos[1] - agent_pos[1])

    def get_closest_task_name(self, available_tasks, agent_pos):
        closest = random.choice(list(available_tasks.keys()))
        occupied = []
        for path in self.token['agents'].values():
            if len(path) == 1:
                occupied.append(path[0])
        dist = self.admissible_heuristic(available_tasks[closest][0], agent_pos)
        for task_name, task in available_tasks.items():
            if self.admissible_heuristic(task[0], agent_pos) < dist and task[0] not in occupied:
                closest = task_name
        return closest

    def get_moving_obstacles_agents(self, agents, time_start):
        obstacles = {}
        for name, path in agents.items():
            if len(path) > time_start and len(path) > 1:
                for i in range(time_start, len(path)):
                    k = i - time_start
                    obstacles[(path[i][0], path[i][1], k)] = name
                    # Mark last element with negative time to later turn it into idle obstacle
                    if i == len(path) - 1:
                        obstacles[(path[i][0], path[i][1], -k)] = name
        return obstacles

    def get_idle_obstacles_agents(self, agents_paths, time_start):
        obstacles = set()
        for path in agents_paths:
            if len(path) == 1:
                obstacles.add((path[0][0], path[0][1]))
            if 1 < len(path) <= time_start:
                obstacles.add((path[-1][0], path[-1][1]))
        return obstacles

    def check_safe_idle(self, agent_pos):
        for task_name, task in self.token['tasks'].items():
            if tuple(task[0]) == tuple(agent_pos) or tuple(task[1]) == tuple(agent_pos):
                return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(agent_pos):
                return False
        return True

    def check_reachable_task_endpoint(self, task, current_agent):
        for task_name, task1 in self.token['tasks'].items():
            if tuple(task1[0]) == tuple(task) or tuple(task1[1]) == tuple(task):
                return False
        for start_goal in self.get_agents_to_tasks_starts_goals():
            if tuple(start_goal) == tuple(task):
                return False
        for agent_name, agent in self.token['agents'].items():
            if task in agent and agent[0] != current_agent:
                # if task in tuple(agent):
                return False
            if agent[0] != current_agent and tuple(task) in self.preempted_locations[agent_name]:
                return False
        return True

    def get_closest_non_task_endpoint(self, agent_pos):
        dist = -1
        res = -1
        for endpoint in self.non_task_endpoints:
            exit = False
            for agent in self.token['agents'].values():
                if len(agent) == 1 and tuple(agent[0]) == endpoint:
                    exit = True
            if not exit and endpoint not in self.token['occupied_non_task_endpoints']:
                if dist == -1:
                    dist = self.admissible_heuristic(endpoint, agent_pos)
                    res = endpoint
                else:
                    tmp = self.admissible_heuristic(endpoint, agent_pos)
                    if tmp < dist:
                        dist = tmp
                        res = endpoint
        if res == -1:
            # #self.print('Error in finding non-task endpoint, is instance well-formed?')
            res = agent_pos
        #  exit(1)
        return res

    def get_preemption_zone(self, location):
        preempted_locations = self.get_preempted_locations()
        zone = []
        if location not in self.starts:
            return []
        for start in self.preemption_zones[tuple(location)]:
            if self.check_reachable_task_endpoint(start, location) and tuple(start) not in preempted_locations:
                zone.append(tuple(start))
        if location in self.starts and tuple(location) not in zone:
            zone.append(tuple(location))
        return list(set(zone))

    def get_preempted_locations(self):
        locations = []
        for zone in self.preempted_locations.values():
            for location in zone:
                locations.append(location)
        return list(locations)

    def get_best_idle_location(self, agent_pos, best_task=None):
        dist = -1
        res = [-1, -1]
        if best_task is not None and tuple(agent_pos) in self.starts and best_task in self.preempted_locations[
            tuple(agent_pos)]:
            # self.print("agent at " + str(agent_pos) + "selects preempted task " + str(best_task))
            return best_task
        for i in range(self.task_distribution.shape[1]):  # self.task_distribution.shape[0]):
            for j in range(self.task_distribution.shape[2]):  # self.task_distribution.shape[0]):
                task = [i, j]
                if task in self.starts and self.check_reachable_task_endpoint(task, agent_pos):
                    x = 0
                    distance = self.admissible_heuristic([i, j], agent_pos) + 1
                    preemption_zone = self.get_preemption_zone(task)
                    for t in range(self.simulation.time + 1,
                                   min(self.simulation.time + int(distance + 1) + self.preemption_duration,
                                       self.task_distribution.shape[3])):
                        for location in preemption_zone:
                            x = x + self.task_distribution[0, location[0], location[1], t]
                    tmp = x / ((distance + self.preemption_duration))
                    if dist == -1:
                        dist = tmp
                        res = [i, j]
                    else:
                        if tmp > dist:
                            dist = tmp
                            res = [i, j]
        if res == [-1, -1]:
            return best_task
        # exit(1)
        if best_task is not None and 1 / (  # len(self.get_preemption_zone(res)) / (
                self.admissible_heuristic(agent_pos, best_task) + 1 + self.preemption_duration) >= dist:
            #  #self.print("agent at " + str(agent_pos) + "selects preempted task " + str(best_task))
            return best_task
        elif best_task is not None:
            True  # self.print("agent at " + str(agent_pos) + "preferred probability of " + str(res) + " to task" + str(best_task))
        return res

    def update_ends(self, agent_pos):
        if tuple(agent_pos) in self.token['path_ends']:
            self.token['path_ends'].remove(tuple(agent_pos))
        if tuple(agent_pos) in self.token['occupied_non_task_endpoints']:
            self.token['occupied_non_task_endpoints'].remove(tuple(agent_pos))

    def get_agents_to_tasks_goals(self):
        goals = set()
        for el in self.token['agents_to_tasks'].values():
            goals.add(tuple(el['goal']))
        return goals

    def get_agents_to_tasks_starts_goals(self):
        starts_goals = set()
        for el in self.token['agents_to_tasks'].values():
            starts_goals.add(tuple(el['goal']))
            starts_goals.add(tuple(el['start']))
        return starts_goals

    def get_completed_tasks(self):
        return self.token['completed_tasks']

    def get_completed_tasks_times(self):
        return self.token['completed_tasks_times']

    def get_token(self):
        return self.token

    def search(self, cbs, agent_name, moving_obstacles_agents):
        path = cbs.search()
        return path

    def go_to_closest_non_task_endpoint(self, agent_name, agent_pos, all_idle_agents, path_modified):
        x = None
        closest_non_task_endpoint = None
        if path_modified:
            closest_non_task_endpoint = self.get_best_idle_location(agent_pos)
            x = closest_non_task_endpoint
        if closest_non_task_endpoint is None:  # or self.simulation.time > 100:
            closest_non_task_endpoint = self.get_closest_non_task_endpoint(agent_pos)
        moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
        idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), 0)
        agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_non_task_endpoint}
        env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents, moving_obstacles_agents,
                          a_star_max_iter=self.a_star_max_iter)
        cbs = CBS(env)
        path_to_non_task_endpoint = self.search(cbs, agent_name, moving_obstacles_agents)
        if not path_to_non_task_endpoint:
            True
            # #self.print("Solution to non-task endpoint not found for agent", agent_name, " instance is not well-formed.")
            # self.deadlock_recovery(agent_name, agent_pos, all_idle_agents, 4)
            # y = self.get_best_idle_location(agent_pos)
        # exit(1)
        else:
            # #self.print('No available task for agent', agent_name, ' moving to safe idling position...')
            self.update_ends(agent_pos)
            if x is not None:
                last_step = path_to_non_task_endpoint[agent_name][-1]
                zone = self.get_preemption_zone([last_step['x'], last_step['y']])
                if agent_pos in self.preemption_zones[tuple([last_step['x'], last_step['y']])] and tuple(
                        agent_pos) not in zone:
                    zone.append(tuple(agent_pos))
                self.preempted_locations[agent_name] = zone
                if len(zone) > 1:
                    True  # self.print(agent_name + " preempted zone " + str(zone))
                self.preemption_status[agent_name] = self.preemption_duration
                self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                self.token['agents_to_tasks'][agent_name] = {'task_name': "test", 'start': agent_pos,
                                                             'goal': closest_non_task_endpoint, 'predicted_cost': 0}

            else:
                self.token['agents_to_tasks'][agent_name] = {'task_name': 'safe_idle', 'start': agent_pos,
                                                             'goal': closest_non_task_endpoint, 'predicted_cost': 0}
                self.token['occupied_non_task_endpoints'].add(tuple(closest_non_task_endpoint))
            self.token['agents'][agent_name] = []
            for el in path_to_non_task_endpoint[agent_name]:
                self.token['agents'][agent_name].append([el['x'], el['y']])
            if not self.token['agents'][agent_name]:
                self.token['agents'][agent_name].append(agent_pos)

    def get_random_close_cell(self, agent_pos, r):
        while True:
            cell = (
                agent_pos[0] + random.choice(range(-r - 1, r + 1)), agent_pos[1] + random.choice(range(-r - 1, r + 1)))
            if cell not in self.obstacles and cell not in self.token['path_ends'] and \
                    cell not in self.token['occupied_non_task_endpoints'] \
                    and cell not in self.get_agents_to_tasks_goals() \
                    and 0 <= cell[0] < self.dimensions[0] and 0 <= cell[1] < self.dimensions[1]:
                return cell

    def deadlock_recovery(self, agent_name, agent_pos, all_idle_agents, r):
        self.token['deadlock_count_per_agent'][agent_name] += 1
        if self.token['deadlock_count_per_agent'][agent_name] >= 5:
            self.token['deadlock_count_per_agent'][agent_name] = 0
            random_close_cell = self.get_random_close_cell(agent_pos, r)
            moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
            idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), 0)
            agent = {'name': agent_name, 'start': agent_pos, 'goal': random_close_cell}
            env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents, moving_obstacles_agents,
                              a_star_max_iter=self.a_star_max_iter)
            cbs = CBS(env)
            path_to_non_task_endpoint = self.search(cbs, agent_name, moving_obstacles_agents)
            if not path_to_non_task_endpoint:
                True
                # #self.print("No solution to deadlock recovery for agent", agent_name, " retrying later.")
            else:
                # Don't consider this a task, so don't add to agents_to_tasks
                # #self.print('Agent', agent_name, 'causing deadlock, moving to safer position...')
                self.update_ends(agent_pos)
                self.token['agents'][agent_name] = []
                for el in path_to_non_task_endpoint[agent_name]:
                    self.token['agents'][agent_name].append([el['x'], el['y']])

    def time_forward(self):
        # Update completed tasks
        for agent_name in self.token['agents']:
            pos = self.simulation.actual_paths[agent_name][-1]
            if agent_name in self.token['agents_to_tasks'] and (pos['x'], pos['y']) == tuple(
                    self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 and self.token['agents_to_tasks'][agent_name][
                'task_name'] != 'safe_idle':
                if self.token['agents_to_tasks'][agent_name]['task_name'] != 'test':
                    self.token['completed_tasks'] = self.token['completed_tasks'] + 1
                    self.token['completed_tasks_times'][
                        self.token['agents_to_tasks'][agent_name]['task_name']] = self.simulation.get_time()
                self.token['agents_to_tasks'].pop(agent_name)
            if agent_name in self.token['agents_to_tasks'] and (pos['x'], pos['y']) == tuple(
                    self.token['agents_to_tasks'][agent_name]['goal']) \
                    and len(self.token['agents'][agent_name]) == 1 and self.token['agents_to_tasks'][agent_name][
                'task_name'] == 'safe_idle':
                self.token['agents_to_tasks'].pop(agent_name)
        # Collect new tasks and assign them, if possible
        for t in self.simulation.get_new_tasks():
            self.token['tasks'][t['task_name']] = [t['start'], t['goal']]
            self.token['start_tasks_times'][t['task_name']] = self.simulation.get_time()
        idle_agents = self.get_idle_agents()
        while len(idle_agents) > 0:
            agent_name = random.choice(list(idle_agents.keys()))

            # agent_name = list(idle_agents.keys())[0]
            # 27.6, 37.24, 25.72
            all_idle_agents = self.token['agents'].copy()
            all_idle_agents.pop(agent_name)
            agent_pos = idle_agents.pop(agent_name)[0]
            if agent_name in self.preempted_locations.keys() and tuple(agent_pos) in self.preempted_locations[
                agent_name]:
                preemption_duration = self.preemption_status[agent_name]
                if preemption_duration == 0:
                    if len(self.preempted_locations[agent_name]) > 1:
                        True  # self.print(agent_name + " removed preemption on " + str(self.preempted_locations[agent_name]))
                    # print("released: ", str(self.preempted_locations[agent_name]))
                    self.preempted_locations[agent_name] = []
                else:
                    self.preemption_status[agent_name] -= 1
            else:
                preemption_duration = 0
            available_tasks = {}
            for task_name, task in self.token['tasks'].items():

                x1 = tuple(task[0]) not in self.token['path_ends'].difference({tuple(agent_pos)})
                x2 = tuple(task[1]) not in self.token['path_ends'].difference({tuple(agent_pos)})
                x3 = tuple(task[0]) not in self.get_agents_to_tasks_goals()
                x4 = tuple(task[1]) not in self.get_agents_to_tasks_goals()
                x5 = ((tuple(task[0]) not in self.get_preempted_locations() and preemption_duration == 0) or
                      (agent_pos in self.starts and tuple(task[0]) in self.preempted_locations[agent_name]))
                if x1 and x2 and x3 and x4 and x5:
                    available_tasks[task_name] = task

            if (len(available_tasks) > 0 or agent_name in self.token['agents_to_tasks']):
                x = None
                if agent_name in self.token['agents_to_tasks']:
                    closest_task_name = self.token['agents_to_tasks'][agent_name]['task_name']

                    closest_task = [self.token['agents_to_tasks'][agent_name]['start'],
                                    self.token['agents_to_tasks'][agent_name]['goal']]
                    x = 0
                else:
                    closest_task_name = self.get_closest_task_name(available_tasks, agent_pos)
                    closest_task = available_tasks[closest_task_name]
                    if preemption_duration == 0 and self.path_1_modified and self.admissible_heuristic(
                            self.get_best_idle_location(agent_pos, closest_task[0]),
                            agent_pos) < self.admissible_heuristic(closest_task[0], agent_pos):
                        self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents, True)
                        x = 0
                if x is None:
                    moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], 0)
                    idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(), 0)
                    agent = {'name': agent_name, 'start': agent_pos, 'goal': closest_task[0]}
                    env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents,
                                      moving_obstacles_agents, a_star_max_iter=self.a_star_max_iter)
                    cbs = CBS(env)
                    path_to_task_start = self.search(cbs, agent_name, moving_obstacles_agents)
                    if not path_to_task_start:
                        True
                        # #self.print("Solution not found to task start for agent", agent_name, " idling at current position...")
                    else:
                        # #self.print("Solution found to task start for agent", agent_name, " searching solution to task goal...")
                        cost1 = env.compute_solution_cost(path_to_task_start)
                        # Use cost - 1 because idle cost is 1
                        moving_obstacles_agents = self.get_moving_obstacles_agents(self.token['agents'], cost1 - 1)
                        idle_obstacles_agents = self.get_idle_obstacles_agents(all_idle_agents.values(),
                                                                               cost1 - 1)
                        agent = {'name': agent_name, 'start': closest_task[0], 'goal': closest_task[1]}
                        env = Environment(self.dimensions, [agent], self.obstacles | idle_obstacles_agents,
                                          moving_obstacles_agents, a_star_max_iter=self.a_star_max_iter)
                        cbs = CBS(env)
                        path_to_task_goal = self.search(cbs, agent_name, moving_obstacles_agents)
                        if not path_to_task_goal:
                            True
                            # #self.print("Solution not found to task goal for agent", agent_name, " idling at current position...")
                        else:
                            # #self.print("Solution found to task goal for agent", agent_name, " doing task...")
                            cost2 = env.compute_solution_cost(path_to_task_goal)
                            if agent_name not in self.token['agents_to_tasks']:
                                self.token['tasks'].pop(closest_task_name)
                                task = available_tasks.pop(closest_task_name)
                            else:
                                task = closest_task
                            last_step = path_to_task_goal[agent_name][-1]
                            self.update_ends(agent_pos)
                            self.preemption_status[agent_name] = 0
                            self.preempted_locations[agent_name] = [tuple([last_step['x'], last_step['y']])]
                            if preemption_duration > 0:
                                True  # self.print(agent_name + "  removed preemption for " )
                            self.token['path_ends'].add(tuple([last_step['x'], last_step['y']]))
                            self.token['agents_to_tasks'][agent_name] = {'task_name': closest_task_name,
                                                                         'start': task[0],
                                                                         'goal': task[1],
                                                                         'predicted_cost': cost1 + cost2}
                            self.token['agents'][agent_name] = []
                            for el in path_to_task_start[agent_name]:
                                self.token['agents'][agent_name].append([el['x'], el['y']])
                            # Don't repeat twice same step
                            self.token['agents'][agent_name] = self.token['agents'][agent_name][:-1]
                            for el in path_to_task_goal[agent_name]:
                                self.token['agents'][agent_name].append([el['x'], el['y']])
            elif not self.path_2_modified and ((not self.path_1_modified and self.check_safe_idle(
                    agent_pos)) or self.check_reachable_task_endpoint(agent_pos, agent_pos)):
                True  # if self.path_1_modified:
                #    self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents, False)
                # #self.print('No available tasks for agent', agent_name, ' idling at current position...')
            elif preemption_duration == 0:
                self.go_to_closest_non_task_endpoint(agent_name, agent_pos, all_idle_agents, self.path_2_modified)

    def print(self, string):
        print("TIME " + str(self.simulation.time) + ": " + string)
