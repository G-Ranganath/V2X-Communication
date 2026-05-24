import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import torch
import torch.nn as nn
import pickle

# =========================
# SETTINGS
# =========================
COMM_RANGE = 150
SEQ_LEN = 5
AI_ENABLED = True   # 🔥 Toggle AI ON/OFF

# =========================
# MODEL
# =========================
class MultiTaskLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 64, batch_first=True)
        self.fc_beam = nn.Linear(64, 32)
        self.fc_action = nn.Linear(64, 1)

    def forward(self, x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        return self.fc_beam(out), self.fc_action(out)

model = MultiTaskLSTM()
model.load_state_dict(torch.load('intersection_multitask_lstm.pth'))
model.eval()

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

print("✅ AI Model Loaded")

# =========================
# CLASSES
# =========================
class RSU:
    def __init__(self,id,x,y):
        self.id=id
        self.x=x
        self.y=y

class Vehicle:
    def __init__(self,id,x,y,speed,dir):
        self.id=id
        self.x=x
        self.y=y
        self.dir=dir
        self.target=speed*(1000/3600)
        self.speed=self.target
        self.history=[]
        self.status="AI START"
        self.prev_rsu=None

    def move(self,dt):
        if self.dir=='E': self.x+=self.speed*dt
        elif self.dir=='W': self.x-=self.speed*dt
        elif self.dir=='N': self.y+=self.speed*dt
        elif self.dir=='S': self.y-=self.speed*dt

    def go(self):
        self.speed=min(self.target,self.speed+3)
        self.status="GO"

    def brake(self):
        self.speed=max(0,self.speed-8)
        self.status="STOP"

# =========================
# ENVIRONMENT
# =========================
rsus=[
    RSU(1,-50,50), RSU(2,50,50),
    RSU(3,-50,-50), RSU(4,50,-50)
]

vehicles=[
    Vehicle(1,-300,-4,90,'E'),
    Vehicle(2,350,4,110,'W'),
    Vehicle(3,4,-250,80,'N'),
    Vehicle(4,-4,400,100,'S')
]

# =========================
# PLOT SETUP
# =========================
fig,ax=plt.subplots(figsize=(10,10))
ax.set_xlim(-250,250)
ax.set_ylim(-250,250)
ax.set_facecolor('#1e1e1e')
ax.set_title("AI V2X Communication System",color='white')

# Roads
ax.fill_between([-250,250],-15,15,color='#333')
ax.fill_betweenx([-250,250],-15,15,color='#333')

# RSU + Coverage
for r in rsus:
    ax.scatter(r.x,r.y,c='cyan',marker='^',s=300,edgecolors='white')
    ax.text(r.x,r.y+12,f"RSU {r.id}",color='cyan',ha='center')
    ax.add_patch(Circle((r.x,r.y),COMM_RANGE,color='cyan',alpha=0.08))

veh_plot=[ax.scatter([],[],c='lime',s=100) for _ in vehicles]
veh_text=[ax.text(0,0,'',color='white') for _ in vehicles]
beam=[ax.plot([],[],lw=2)[0] for _ in vehicles]

# =========================
# UPDATE LOOP
# =========================
def update(frame):
    dt=0.1

    for i,v in enumerate(vehicles):

        dist_center=np.hypot(v.x,v.y)

        dist_closest=min([
            np.hypot(v.x-o.x,v.y-o.y)
            for o in vehicles if o!=v
        ]+[999])

        t_brake=max(0,(dist_center-20)/(v.speed+0.1))

        state=[v.x,v.y,v.speed,dist_center,dist_closest,t_brake]
        v.history.append(state)
        if len(v.history)>SEQ_LEN:
            v.history.pop(0)

        # ================= AI =================
        if len(v.history)==SEQ_LEN:

            if AI_ENABLED:
                seq=scaler.transform(v.history)
                seq=torch.tensor(seq,dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    _,action=model(seq)

                if torch.sigmoid(action)>0.5:
                    v.go()
                else:
                    v.brake()

            else:
                # No AI → unstable behavior
                if frame % 10 < 5:
                    v.go()
                else:
                    v.brake()

            # ================= RSU HANDOVER =================
            rsu=min(rsus,key=lambda r:np.hypot(v.x-r.x,v.y-r.y))

            if v.prev_rsu and v.prev_rsu!=rsu.id:
                print(f"🔄 Vehicle {v.id} handover: RSU {v.prev_rsu} → RSU {rsu.id}")

            v.prev_rsu=rsu.id

            dist=np.hypot(v.x-rsu.x,v.y-rsu.y)

            # ================= SNR =================
            snr=max(0,30-0.12*dist)

            # INTERFERENCE
            if dist_closest < 40:
                snr -= 5

            # ================= LATENCY =================
            latency=2+(dist/50)

            # ================= PACKET LOSS =================
            packet_loss=max(0,(20-snr)*2)

            # ================= BEAM =================
            if dist<COMM_RANGE:
                beam[i].set_data([rsu.x,v.x],[rsu.y,v.y])

                strength=1-dist/COMM_RANGE
                beam[i].set_color((1,strength,0))
            else:
                beam[i].set_data([],[])

        else:
            snr=0
            latency=0
            packet_loss=0
            beam[i].set_data([],[])

        # Move vehicle
        v.move(dt)
        veh_plot[i].set_offsets([v.x,v.y])

        # HUD
        veh_text[i].set_position((v.x+5,v.y+5))
        veh_text[i].set_text(
            f"V{v.id}\n"
            f"{int(v.speed*3.6)} km/h\n"
            f"SNR: {snr:.1f} dB\n"
            f"Latency: {latency:.1f} ms\n"
            f"Loss: {packet_loss:.1f}%\n"
            f"{v.status}"
        )

    return veh_plot + veh_text + beam

# =========================
# RUN
# =========================
ani=FuncAnimation(fig,update,frames=800,interval=40,blit=True)
plt.show()
