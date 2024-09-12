from collections import defaultdict
from functools import partial as bind

import embodied
import numpy as np
import os
import re

from PIL import Image

actual_trajectory_actions = None
actual_trajectory_images = None
decoded_fwd_images = None

def what_if_validation(make_agent, make_env, make_logger, args):
  assert args.from_checkpoint

  agent = make_agent()
  logger = make_logger()

  logdir = embodied.Path(args.logdir)
  logdir.mkdir()
  print('Logdir', logdir)
  step = logger.step
  usage = embodied.Usage(**args.usage)
  agg = embodied.Agg()
  epstats = embodied.Agg()
  episodes = defaultdict(embodied.Agg)
  should_log = embodied.when.Clock(args.log_every)
  policy_fps = embodied.FPS()

  def cache_actual_choices(tran, worker):
    global actual_trajectory_actions 
    global actual_trajectory_images
    actual_trajectory_actions =  tran['action'] if actual_trajectory_actions is None else np.append(actual_trajectory_actions, tran['action'])
    actual_image = tran['image'][np.newaxis, :, :, :]
    actual_trajectory_images = actual_image if actual_trajectory_images is None else np.concatenate([actual_trajectory_images, actual_image], axis=0)

  def rev_step(obs, alt_action, dup_carry):
    return agent.rev_step(obs, [{'action': np.array([alt_action], dtype=np.int32)}], dup_carry)['image'][0]

  """ get_comparison_image generates and hence compares the following:
      1. actual_trajectory_image: The actual image in the observation, at a given time step
      2. decoded_fwd_image: The image as predicted by the (forward) WM, if one were to do noop on the previous observation
          - Aim is to make the Rev WM predictions to get as close to the actual observation as possible. 
            But until that happens, we need a realistic baseline to ensure that the Rev WM is actually 
            learning and getting better. So we print the fwd WM results as well as an intermediate 
            baseline.
      3. predicted_rev_images: We run the rev WM on the outputs of alt_actions to get what the rev WM thinks the previous
        state would have looked like. This is the same image as what we have printed in 1 and 2. This tells us how close to 
        truth the predictions of the rev WM are.
  """
  def get_comparison_image(actual_trajectory_image, decoded_fwd_image, predicted_rev_images):    
    # Both fwd and rev WM predicted images are normalized, and hence need this edit
    decoded_fwd_image = np.clip(255 * decoded_fwd_image, 0, 255).astype(np.uint8)
    combined_pred_images = np.concatenate([image for image in predicted_rev_images], axis=1)
    combined_pred_images = np.clip(255 * combined_pred_images, 0, 255).astype(np.uint8)

    return np.concatenate([actual_trajectory_image, decoded_fwd_image, combined_pred_images], axis=1)

  @embodied.timer.section('log_step')
  def log_step(tran, worker):
    episode = episodes[worker]
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')

    if tran['is_first']:
      episode.reset()

    if worker < args.log_video_streams:
      for key in args.log_keys_video:
        if key in tran:
          episode.add(f'policy_{key}', tran[key], agg='stack')
    for key, value in tran.items():
      if re.match(args.log_keys_sum, key):
        episode.add(key, value, agg='sum')
      if re.match(args.log_keys_avg, key):
        episode.add(key, value, agg='avg')
      if re.match(args.log_keys_max, key):
        episode.add(key, value, agg='max')

    if tran['is_last']:
      result = episode.result()

      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length') - 1,
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  fns = [bind(make_env, i, needs_episode_reset = True) for i in range(args.num_envs)]
  driver = embodied.Driver(fns, args.driver_parallel)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(log_step)
  driver.on_step(cache_actual_choices)
 
  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation')
  global decoded_fwd_images
  policy = lambda *args: agent.policy(*args, mode='eval')
  driver.reset(agent.init_policy)
  # Make the agent go through a regular based actions selected by its own policy
  while step < args.steps:
    fwd_images = driver(policy, steps=10)
    decoded_fwd_images = fwd_images if decoded_fwd_images is None else np.concatenate([decoded_fwd_images, fwd_images], axis=0)
    if should_log(step):
      logger.add(agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(embodied.timer.stats(), prefix='timer')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.write()

  # Make agent take what-if actions
  driver.callbacks = []
  alt_actions = np.array([1, 2, 3, 4]) # Avoid repeat actions
  predicted_rev_images = np.zeros((len(alt_actions), 64, 64, 3))

  images_folder = 'test_images/'
  if os.path.exists(images_folder) is False:
    os.mkdir(images_folder)

  # For every state in the original trajectory, make the agent take 4 other actions and cache the generated transitions
  # The four other actions, in order are: 'move_left','move_right','move_up','move_down'
  for current_step in range(len(actual_trajectory_actions)): 
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
        step, episode, _ = driver._perform_action([{'action': np.array([actual_trajectory_actions[step]], dtype=np.int32), 'reset': np.array([False], dtype=bool)}], step, episode)
      print('what_if_validation: Historical actions complete')

      # take the alt action
      print('what_if_validation: Taking alt action ', alt_action)
      _, _, obs = driver._perform_action([{'action': np.array([alt_action], dtype=np.int32), 'reset': np.array([False], dtype=bool)}], step, episode)
      print('what_if_validation: Alt action complete')

      # get the predicted state according to rev WM
      print('what_if_validation: rev_step begins')
      predicted_rev_images[np.argwhere(alt_actions == alt_action)[0][0]] = rev_step(obs, alt_action, driver.dup_carry)
      print('what_if_validation: rev_step ends')

    combined_image = get_comparison_image(actual_trajectory_images[current_step], decoded_fwd_images[current_step], predicted_rev_images )
    image = Image.fromarray(combined_image, 'RGB')

    image.save(images_folder + str(current_step) + '_combined_image.png')

  logger.close()
