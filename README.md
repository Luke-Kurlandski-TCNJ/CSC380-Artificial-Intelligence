Instructions for use on TCNJ HPC, ELSA:
1. > ssh userid@elsa.hpc.tcnj.edu
2. > git clone https://github.com/Luke-Kurlandski-TCNJ/CSC380-Artificial-Intelligence.git
3. > cd CSC380-Artificial-Intelligence
4. > module add python/3.7.5
5. > python3 -m venv env
6. > source env/bin/activate
7. > pip install -r requirements.txt

Every time you start a new session with ELSA, you will have to perform steps 4 & 5.

Consider this CSC380-Artificial-Intelligence the root directory. To run the Nth homework assignment:
> python3 HWN/main.py

Content of this repository:
- HWN/ -- directory containing particular homework assignment
- utils.py -- general purpose utility functions used through the semester