case_name="105" 
gpu_id=0
surface_weight=0.1

nohup python exp_runner.py --mode train --conf ./confs/womask.conf \
--case "public_data/scan${case_name}" --type dtu --surface_weight ${surface_weight}\
--gpu ${gpu_id} > "logs/log_dtu_${case_name}" 2>&1

nohup python lvis.py --mode train --conf ./confs/womask.conf \
--case "public_data/scan${case_name}" --type dtu \
--gpu ${gpu_id} > "logs/log_dtu_${case_name}" 2>&1

nohup python mateIllu.py --mode train --conf ./confs/womask.conf \
--case "public_data/scan${case_name}" --type dtu \
--gpu ${gpu_id} > "logs/log_dtu_${case_name}" 2>&1

# clean mesh and eval mesh using scripts clean_mesh_pose.py and eval_mesh.py

