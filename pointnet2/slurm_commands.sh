srun -p content_generation --gres=gpu:8 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task=128 --job-name
srun -p content_generation --gres=gpu:4 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task=128 --job-name
srun -p content_generation --gres=gpu:2 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task=64 --job-name
srun -p content_generation --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --kill-on-bad-exit=1 --cpus-per-task=32 --job-name
srun -p content_generation --ntasks=1 --ntasks-per-node=1 --cpus-per-task=64 --job-name
