import subprocess


scene_seq = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

def generate_obs(command_path, dataset_path, segment_save_path, track_save_path, model_path='None', file=None):
  result_all = []
  # segment or tracking
  
  for seq in scene_seq:
      if model_path is not 'None': # for track
        result = subprocess.Popen(['python', command_path, '--seq', str(seq), '--save_path', str(track_save_path), '--dataset_path', segment_save_path, '--model_path_backbone', model_path[0], '--model_path_head', model_path[1]], stdout = file )
      else: # for clustering
        result = subprocess.Popen(['python', command_path, '--seq', str(seq), '--save_path', str(segment_save_path), '--dataset_path', dataset_path], stdout = file )
      result_all.append(result)
      
  while(True):
    count = 0
    for i in range(len(result_all)):
      if result_all[i].poll() is None:
        count = 0 
        break;
      else:
        count+=1
    if count == len(result_all):
      break;

