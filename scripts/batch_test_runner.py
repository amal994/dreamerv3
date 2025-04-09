import os
import subprocess
import numpy as np

agent_ids = np.arange(4)
scene_ids = np.arange(4)
num_steps = [100]*4 # agent runs until it completes an episode or these many steps, whichever happens first
scripts_to_run = ['in_dist_imagination']

# Base folder paths
log_base_folder = 'logs'
test_images_base_folder = 'test_images'
exp_set = 'test_run_abc'
# Ends

for script in scripts_to_run:
    for agent_id in agent_ids:
        for scene_id in scene_ids:
            log_folder = os.path.join(log_base_folder, exp_set, script, 'agent_' + str(agent_id), 'scene_' + str(scene_id))
            test_images_folder =  os.path.join(test_images_base_folder, exp_set, 'agent_' + str(agent_id), 'scene_' + str(scene_id))
            
            os.makedirs(log_folder, exist_ok=True)
            os.makedirs(test_images_folder, exist_ok=True)
            
            print('Running test script ', script, ' with agent ', agent_id, ' on scene ', scene_id)
            command = 'python dreamerv3/main.py'
            command = command + ' --logdir ' + log_folder
            command = command + ' --configs crafter size25m'
            command = command + ' --run.script ' + script
            # Edit the checkpoint based on its location
            command = command + ' --run.from_checkpoint /mnt/hdd/msingh365/checkpoints/skynet/4ags_ncr_correct_recipes/scenario' + str(agent_id)+'/checkpoint.ckpt'
            command = command + ' --run.steps ' +  str(num_steps[scene_id])
            command = command + ' --run.env_index ' + str(scene_id)
            command = command + ' --run.test_images_folder ' + test_images_folder
            command = command + ' --run.scene_label static_size_15_abberation_set_' + str(agent_id)

            log_file_path = os.path.join(log_folder, script + 'agent_' + str(agent_id) + '_scene_' + str(scene_id) + '.log')

            print('RUNNING COMMAND ', command)
            print('LOG FILE : ', log_file_path)
            print('TEST IMAGES LOCATION : ', test_images_folder)
            log_file = open(log_file_path, 'w')
            ret = subprocess.run(command.split(), stdout=log_file)
            log_file.close()