dataset_name=banc2
shared_dataset=/data/$dataset_name
dataset_root=public_data/$dataset_name
database_path=$dataset_root/database.db
transforms_path=$dataset_root/transforms.json
image_path=$dataset_root/image/
mask_path=$dataset_root/mask/
sparse_path=$dataset_root/sparse
mkdir -p $sparse_path

cp -r $shared_dataset $dataset_root

# cree le fichier database.db
colmap feature_extractor --database_path $database_path --image_path $image_path --ImageReader.mask_path $mask_path --ImageReader.single_camera 1

# ?
#colmap sequential_matcher --database_path $database_path --SequentialMatching.overlap 15
# ou ceci avec un petit dataset : (?)
colmap exhaustive_matcher --database_path $database_path


# popule le dossier sparse
colmap mapper --database_path $database_path --image_path $image_path --output_path $sparse_path

# ???
colmap model_converter --input_path $sparse_path/0 --output_path $sparse_path/0 --output_type TXT

# converti le modele de colmap en nerf
python colmap2nerf.py --images $image_path --text $sparse_path/0 --out $transforms_path

# converti le fichier transforms.json en cameras_sphere.npz
python transforms2cameras_sphere.py --transforms_path $dataset_root/transforms.json --cameras_sphere_path $dataset_root/cameras_sphere.npz

./delete_missing_images.sh


python exp_runner.py --conf confs/wmask.conf --gpu 0 --case $dataset_name
