This tool allows the user to buld ESPs either from precomputed QM calcs or a machine learning charge model. 

Installation
==============

1. Create a conda environment with the required packages with the ENV.yml available

`conda create -n esp_vis --file ENV.yml`

2. Activate this environment. Now in a separate directory, clone the molesp repo

`git clone git@github.com:SimonBoothroyd/molesp.git`

3. Install the gui in molesp by entering the molesp directory and running

`pip install -e .`

then

`python setup.py build_gui`

then

`python setup.py install`

4. Now we need to install the ML charge models in order to produce ESPs quickly.
First clone this repo

`git clone https://github.com/bismuthadams1/nagl-mbis`

then install a key dependecy with 

`pip install git+https://github.com/SimonBoothroyd/nagl.git@main`

finally, in the nagl-mbis repo run

`pip install -e . --no-build-isolation`

update the molesp env 


ESPs From QM Calcs Install
==========================
 
5. In order to produce ESPs from QM calcs, we need a bespoke .db file generated in 
my code Chargecraft. Charge craft can be cloned here

TODO
``

There are two modes that you can use this visualizer:

1. Visualize an ESP with a charge model

2. Visualize an ESP with precomputed QM values

Running The Program with the Charge Models
==========================================

To use charge models you must first follow the install instructions from this repo and install the ChargeAPI in this esp_vis environment:

https://github.com/bismuthadams1/ChargeAPI


For this mode, we need to go the 'esp_visualize_esp.py' file. Add your chosen .sdf file and then 
run the code. A GUI will then be available at http://localhost:8000.



