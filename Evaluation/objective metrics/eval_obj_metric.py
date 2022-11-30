
import os
import muspy
from datetime import datetime

path = r"./samples/survey_samples/myemo/"
filenames = os.listdir(path)

num_files = len(filenames)
PR = 0        # 1. pitch_range
NPC = 0       # 2. n_pitch_classes_used
POLY = 0      # 3. polyphony
SC = 0        # 4. scale consistency

i = 0
for file in filenames:
    i = i+1
    mus_obj = muspy.read_midi(path + file)
    
    
    PR = PR + muspy.pitch_range(mus_obj)
    NPC = NPC + muspy.n_pitch_classes_used(mus_obj)
    POLY = POLY + muspy.polyphony(mus_obj)
    SC = SC + muspy.scale_consistency(mus_obj)

PR_avg = PR/num_files
NPC_avg = NPC/num_files
POLY_avg = POLY/num_files
SC_avg = SC/num_files

time = datetime.now()

with open(r"./objective_result.txt", "a") as o:
    o.write(str(time)+'\n')
    o.write("folder: "+path+'\n')
    o.write("PR   = "+str(PR_avg)+'\n')
    o.write("NPC  = "+str(NPC_avg)+'\n')
    o.write("POLY = "+str(POLY_avg)+'\n')
    o.write("SC   = "+str(SC_avg)+'\n\n')
