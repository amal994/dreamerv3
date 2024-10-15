from functools import partial as bind

import embodied
import numpy as np

from . import run_utils

'''
in_dist_imagination
validates the correctness of multi-step in-distribution rev WM predictions

Steps are as follows: 
  1. We make the trained agent go through a regular test run based on its
      learned policy - we cache the state and action information for every
      transition that takes place
  2. Reset the environment
  3. Starting from the final state, we go in reverse and feed the final state
     and the full sequence of actions (reversed) to the rev WM. We now ask the
     rev WM to predict the whole trajectory that preceded.
     Note: Latents for carry are only updated based on rev WM's imagined states. 
     Code for this can be seen in Agent::rev_step_in_one_go(...)
'''
def in_dist_imagination(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  def rev_step_in_one_go(obs, actions, dup_carry):
    formatted_actions = [{'action': np.array(actions, dtype=np.int32)}]
    output_images = agent.rev_step_in_one_go(obs, formatted_actions, dup_carry)
    return output_images['image']

  agent = make_agent()
  experiment_tracker = run_utils.ExperimentTracker(args)
  experiment_tracker.setup_tracking(make_logger)
  trajectory_cache = run_utils.TrajectoryCache(['image', 'reward', 'is_first', 'is_last', 'is_terminal', 'log_reward', 'action'])
  image_util = run_utils.ImageUtil(base_folder=args.test_images_folder, experiment_label='in_dist_imagination')

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
  decoded_fwd_images = None
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  # Make the agent go through a regular based actions selected by its own policy
  while experiment_tracker.step < args.steps:
    fwd_images = driver(policy, steps=10)
    decoded_fwd_images = fwd_images if decoded_fwd_images is None else np.concatenate([decoded_fwd_images, fwd_images], axis=0)
    experiment_tracker.log_stats()
  
  num_recorded_trans = trajectory_cache.get_cache_count()
  final_obs = trajectory_cache.get_obs_at(num_recorded_trans - 1)
  del final_obs['action']

  actions_of_interest = trajectory_cache.get_actions()[::-1][1:]
  decoded_rev_images = rev_step_in_one_go(final_obs, actions_of_interest, driver.dup_carry)

  for i in range(len(decoded_rev_images)):
    image_util.print_all_images(actual_image=trajectory_cache.get_image_at(num_recorded_trans - 2 - i),
                            fwd_pred_image=decoded_fwd_images[num_recorded_trans - 2 - i], 
                            rev_pred_images=decoded_rev_images[i][np.newaxis, :], 
                            name_prefix=str(num_recorded_trans - 2 - i))

  experiment_tracker.experiment_complete()