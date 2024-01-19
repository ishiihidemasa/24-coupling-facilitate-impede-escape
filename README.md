Data and scripts for ``Diffusive coupling facilitates and impedes noise-induced escape in interacting bistable elements'' by H Ishii and H Kori.

# How to run the scripts
You can reproduce the python environment with necessary packages using either Docker (`Dockerfile`) or conda (`CON_gcbistable-met.yml`).
- With VS Code ``Dev Containers'' extension and Docker Desktop
  - Simply download this repository
  - Open the directory as container, using `Dev Containers: Open Folder in Container...` command.
    - You may need to reload the window after container is built.
  - You can run the scripts in the container.
- With Docker Desktop
  - Use `Dockerfile` to reproduce the environment.
- With conda
  - Use `CON_gcbistable-met.yml` to reproduce the environment.
    - i.e. `conda env create -f CON_gcbistable-met.yml`
   
# How to reproduce the figures
Run the scripts named `figX_somthing.py`.  
Most of the necessary data are stored in `data` directory, and loaded by the scripts.  
They have flag variables `if_show` and `if_save`. Change their values if necessary.

# How to run calculations
Results stored in `data` directory were calculated using the scripts named `calc_something.py`.  
You can use them to reproduce the stored data and to generate more results.  
Note however that we ran some of the scripts on HPC with torque (job scheduler) and used `pack_results.py` to integrated results into one `.npz` file.  
If you have any question about conducting calculations, please get in touch with me (ISHII Hidemasa).
