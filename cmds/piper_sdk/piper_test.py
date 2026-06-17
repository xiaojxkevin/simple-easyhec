import time
from piper_sdk import *

can_list = ["can_right_main","can_right_main","can_left_main","can_left_slave"]
mit_mode = False

def go_to_zero(can_name):
    piper = C_PiperInterface(can_name=can_name, judge_flag=True)
    piper.MasterSlaveConfig(0xFC, 0, 0, 0)
    piper.ConnectPort()
    
    while( not piper.EnablePiper()):
        time.sleep(0.01)
    factor = 57295.7795 #1000*180/3.1415926
    position = [0,0,0,0,0,0,0]
    
    joint_0 = round(position[0]*factor)
    joint_1 = round(position[1]*factor)
    joint_2 = round(position[2]*factor)
    joint_3 = round(position[3]*factor)
    joint_4 = round(position[4]*factor)
    joint_5 = round(position[5]*factor)
    joint_6 = round(position[6]*1000*1000)
    piper.ModeCtrl(0x01, 0x01, 30, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    time.sleep(5)
    # piper.DisablePiper()
    piper.DisconnectPort()

if __name__ == "__main__":
    for can_name in can_list:
        print(f"{can_name} go to zero.")
        go_to_zero(can_name)
    
    