from functools import partial as bind

import embodied
import numpy as np

from . import run_utils

def what_if_validation(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  def rev_step(obs, alt_action, dup_carry):
    return agent.rev_step(obs, [{'action': np.array([alt_action], dtype=np.int32)}], dup_carry)['image'][0]

  agent = make_agent()
  experiment_tracker = run_utils.ExperimentTracker(args)
  experiment_tracker.setup_tracking(make_logger)
  trajectory_cache = run_utils.TrajectoryCache(['action', 'image'])
  image_util = run_utils.ImageUtil(base_folder=args.test_images_folder, experiment_label='what_if_validation')

  fns = [bind(make_env, i, needs_episode_reset = True) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel)
  driver.on_step(lambda tran, _: experiment_tracker.step.increment())
  driver.on_step(lambda tran, _: experiment_tracker.policy_fps.step())
  driver.on_step(experiment_tracker.log_step)
  driver.on_step(trajectory_cache.cache_partial_obs)
 
  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation')
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  # Make the agent go through a regular run based on actions selected by its own policy
  decoded_fwd_images = None
  while experiment_tracker.step < args.steps:
    fwd_images = driver(policy, steps=10)
    decoded_fwd_images = fwd_images if decoded_fwd_images is None else np.concatenate([decoded_fwd_images, fwd_images], axis=0)
    experiment_tracker.log_stats()

  # Make agent take what-if actions
  driver.callbacks = []
  alt_actions = np.array([1, 2, 3, 4]) # Avoid repeat actions
  predicted_rev_images = np.zeros((len(alt_actions), 64, 64, 3))

  # For every state in the original trajectory, make the agent take 4 other actions and cache the generated transitions
  # The four other actions, in order are: 'move_left','move_right','move_up','move_down'
  for current_step in range(0, len(trajectory_cache.get_actions())): 
    # for every selected different action
    for alt_action in alt_actions:
      # reset environment
      print('what_if_validation: Env resets for current_step ', current_step)
      driver.reset(agent.init_policy)
      driver._perform_action([{'action': np.array([0], dtype=np.int32), 'reset': np.array([True], dtype=bool)}], 0, 0) # Resets Env

      # take the same actions as the original trajectory, until the current transition
      step, episode = 0, 0
      print('what_if_validation: Taking historical actions')
      while(step < current_step):
        step, episode, _ = driver._perform_action([{ 'action': np.array([trajectory_cache.get_action_at(step)], dtype=np.int32), 
                                                     'reset': np.array([False], dtype=bool)
                                                  }], 
                                                  step, 
                                                  episode)
      print('what_if_validation: Historical actions complete')

      # take the alt action
      print('what_if_validation: Taking alt action ', alt_action)
      _, _, obs = driver._perform_action([{'action': np.array([alt_action], dtype=np.int32), 'reset': np.array([False], dtype=bool)}], step, episode)
      print('what_if_validation: Alt action complete')

      # get the predicted state according to rev WM
      print('what_if_validation: rev_step begins')
      predicted_rev_images[np.argwhere(alt_actions == alt_action)[0][0]] = rev_step(obs, alt_action, driver.dup_carry)
      print('what_if_validation: rev_step ends')

      image_util.print_all_images(actual_image=trajectory_cache.get_image_at(current_step),
                                  fwd_pred_image=decoded_fwd_images[current_step], 
                                  rev_pred_images=predicted_rev_images, 
                                  name_prefix=str(current_step))

  experiment_tracker.experiment_complete()