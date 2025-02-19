import gym

class Coffee_rewards_wrapper: #(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """
    completion_reward = 5

    rewards_for_achievements = [
        {   'collect_coffee_bean': 1,
            'collect_milk': 1,
            'collect_sugar_cube': 1,
            'collect_spice': 1,
            'collect_spinach': 0,
            'collect_hot_sauce': 0,
            'collect_drink': 0,
            'make_boiled_milk': 1,
            'make_roasted_coffee_bean': 1,
            'make_coffee_powder': 1,
            'make_coffee': 10,
            'collect_chocolate_bar': 0,
            'collect_coffee_powder': 0,
            'make_boiled_water': 0,
            'make_lava': 0
        }, 
        {   'collect_coffee_bean': 1,
            'collect_milk': 1,
            'collect_sugar_cube': 0,
            'collect_spice': 0,
            'collect_spinach': 0,
            'collect_hot_sauce': 0,
            'collect_drink': 0,
            'make_boiled_milk': 1,
            'make_roasted_coffee_bean': 1,
            'make_coffee_powder': 1,
            'make_coffee': 10,
            'collect_chocolate_bar': 1,
            'collect_coffee_powder': 0,
            'make_boiled_water': 0,
            'make_lava': 0
        }, 
        {   'collect_coffee_bean': 1,
            'collect_milk': 0,
            'collect_sugar_cube': 0,
            'collect_spice': 0,
            'collect_spinach': 0,
            'collect_hot_sauce': 1,
            'collect_drink': 1,
            'make_boiled_milk': 0,
            'make_roasted_coffee_bean': 1,
            'make_coffee_powder': 1,
            'make_coffee': 10,
            'collect_chocolate_bar': 0,
            'collect_coffee_powder': 0,
            'make_boiled_water': 1,
            'make_lava': 1
        }, 
        {   'collect_coffee_bean': 0,
            'collect_milk': 1,
            'collect_sugar_cube': 1,
            'collect_spice': 0,
            'collect_spinach': 0,
            'collect_hot_sauce': 0,
            'collect_drink': 0,
            'make_boiled_milk': 0,
            'make_roasted_coffee_bean': 0,
            'make_coffee_powder': 0,
            'make_coffee': 10,
            'collect_chocolate_bar': 0,
            'collect_coffee_powder': 1,
            'make_boiled_water': 0,
            'make_lava': 0
        }, 
    ]

    achievement_reward_limit = {
        'collect_coffee_bean': 1,
        'collect_milk': 1,
        'collect_sugar_cube': 1,
        'collect_spice': 1,
        'collect_hot_sauce': 1,
        'collect_chocolate_bar': 1,
        'collect_coffee_powder': 1,
        'make_boiled_milk': 1,
        'make_roasted_coffee_bean': 1,
        'make_coffee_powder': 1,
        'make_coffee': 1,
        'make_boiled_water': 1,
        'make_lava': 1,
        'collect_spinach': 1,
        'collect_drink': 1
    }

    def __init__(self, env, env_label, required_achievements = ['make_coffee'], max_steps = 500):
        print('Coffee_rewards_wrapper::__init__ env_label = ', env_label, ', required_achievements = ', required_achievements, ', max_steps = ', max_steps)
        self.env = env
        print('Coffee_rewards_wrapper::__init__::env recipe_id = ', self.env._recipe_id)
        self.positive_achievements = {achievement: 0 for (achievement, reward) in self.rewards_for_achievements[self.env._recipe_id].items() if reward > 0}
        self.first_step_achievement = {achievement: -1 for achievement in self.rewards_for_achievements[self.env._recipe_id].keys() }
        self.max_count = max_steps
        self.required_achievements = required_achievements

        self.prev_achievements = None
        self.step_count = 0

        self.label = env_label

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return getattr(self.env, name)


    def reset(self, **kwargs):
        self.prev_achievements = None
        self.step_count = 0
        self.first_step_achievement = {achievement: -1 for achievement in self.rewards_for_achievements[self.env._recipe_id].keys() }
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        obs, life_reward, done, info = self.env.step(action)
        info.update(env_label=self.label)
        info.update(step_count=self.step_count)
        info.update(life_reward=life_reward)

        achievement_reward = 0
        info.update(current_achievements=None)
        info.update(achievement_reward=None)

        current_achievement_reward_map = {'life_maintenance': life_reward}

        if len(info['achievements']) > 0:
            current_achievements = self.get_current_achievements(info['achievements'], self.prev_achievements)
            for i in range(len(current_achievements)):
                current_ach_reward = self.rewards_for_achievements[self.env._recipe_id][current_achievements[i]]
                current_achievement_reward_map[current_achievements[i]] = current_ach_reward
                achievement_reward += current_ach_reward
            info.update(current_achievement_reward_map = current_achievement_reward_map)
            info.update(current_achievements = current_achievements)
            info.update(achievement_reward = achievement_reward)

        reward = life_reward + achievement_reward
        self.prev_achievements = info['achievements'].copy()

        if self.are_achievements_complete():
            print('Episode is done because achievements are complete')
            done = True
            completion_reward = self.completion_reward*(1 - 0.9*((self.step_count)/self.max_count))
            reward += completion_reward
            # Episode completion happens when diamond is obtained, so to the agent this reward is being given for attaining the diamond
            info['current_achievement_reward_map']['make_coffee'] += completion_reward
        elif self.step_count >= self.max_count:
            print('Episode is done because max steps have been reached')
            info["TimeLimit.truncated"] = True
            done = True

        return obs, reward, done, info

    def is_achievement_reward_limit_reached(self, achievement):
        if achievement not in self.achievement_reward_limit: 
            return False

        if self.prev_achievements is None: 
            return False
        
        return self.prev_achievements[achievement] >= self.achievement_reward_limit[achievement]

    def are_achievements_complete(self):
        if self.prev_achievements is None:
            return False
        required_achievement_list = self.positive_achievements.keys() if self.required_achievements is None else self.required_achievements
        for achievement in required_achievement_list:
            if self.prev_achievements[achievement] < 1:
                return False
        return True

    def get_current_achievements(self, unlocked_list, prev_achievements):
        current_achievements = []
        for achievement in unlocked_list.keys():
            if unlocked_list[achievement] > 0:
                prev_count = 0
                if prev_achievements is not None:
                    prev_count = prev_achievements[achievement]
                if unlocked_list[achievement] > prev_count:
                    if not (self.is_achievement_reward_limit_reached(achievement)):
                        current_achievements.append(achievement)
        return current_achievements
