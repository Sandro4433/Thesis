# Franka Startup & Execution Guide

## Start Franka
- Plug in the Franka controller and turn it on.
- The Franka should blink yellow.
- When it stops blinking, press the button once and release it.

## Start the PC
- During boot, use the arrow keys to select **Advanced options**.
- Select **rt-20_rt-20**.
- The PC should boot normally.
- Login Password: FRANKAEMIKA

## Connect to Franka
- Open **Chrome** and go to `192.168.1.100`.
- The Franka web interface should open.
- On the right side, click the **lock icon** to unlock.
- Click the **three lines** (top right) and select **Activate FCI**.

## Start Simulation / Path Planning
- Open **File Explorer**.
- Go to **Home → Anleitung**.
- Copy the first two commands:
  ```bash
  source ...
  rosrun ... gripper:=true
  ```
- Open a terminal and paste both commands.
- **RViz should open**, and the robot should appear in the simulation.

## Start Python Program
- Press the **Windows key**, search for **Visual Studio**, and open it.
- Open `main1_withrobotClass.py`.
- To connect to the PLC:
  - Comment **line 32**
  - Uncomment **line 31** (`plc = ...`)
- Run the Python script.

## Important Notes
- The **terminal must be running** before starting the Python script.
- **FCI must be activated**.
- If the terminal needs a restart:
  - Press `Ctrl + C`
  - Press `Arrow Up`
  - Press `Enter` to rerun the last command
