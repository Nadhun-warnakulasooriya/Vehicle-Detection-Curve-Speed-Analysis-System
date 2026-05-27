import cv2, time, torch, math, threading, os
import numpy as np
from ultralytics import YOLO
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import socket
import json

# ═══════════════════════════════════════════════════════════
#  DEFAULT PATHS & NETWORK
# ═══════════════════════════════════════════════════════════

DEFAULT_MODEL = r'C:\Final project\runs\vehicle_detection_v1\weights\best.pt'
DEFAULT_VIDEO = r'C:\Users\Nadhu\Downloads\stock-footage-uttarakhand-in-may-the-traffic-on-a-curvy-road-surrounded-by-mountains-under-blue-sky.mp4'
FOCAL_LENGTH  = None

# Target ESP32 config for hardware linkage
ESP32_IP = "192.168.1.100"  # <-- Update this with your ESP32's Local IP
ESP32_PORT = 4210
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ═══════════════════════════════════════════════════════════
#  VEHICLE DATA TABLES
# ═══════════════════════════════════════════════════════════

CLASS_NAMES = [
    'Tree weeler','bicycle','bus','car','minitruck',
    'motorbike','riksha','truck','van',
]
REAL_HEIGHTS = {
    'Tree weeler':1.6,'bicycle':1.1,'bus':3.5,'car':1.5,'minitruck':2.0,
    'motorbike':1.2,'riksha':1.6,'truck':3.8,'van':1.9,
}
VEHICLE_MASS = {
    'Tree weeler':400,'bicycle':90,'bus':12000,'car':1500,'minitruck':3500,
    'motorbike':250,'riksha':400,'truck':15000,'van':2200,
}
SAFETY_FACTOR = {
    'Tree weeler':0.80,'bicycle':1.00,'bus':0.75,'car':1.00,'minitruck':0.85,
    'motorbike':1.00,'riksha':0.80,'truck':0.75,'van':0.90,
}
CLASS_COLORS_BGR = {
    'Tree weeler':(255,180,50),'bicycle':(50,220,255),'bus':(50,50,255),
    'car':(50,200,50),'minitruck':(255,100,200),'motorbike':(0,165,255),
    'riksha':(180,50,255),'truck':(0,80,255),'van':(255,220,0),
}
ROAD_CONDITIONS = {
    'Dry Asphalt':0.70,'Wet Asphalt':0.40,'Gravel':0.30,'Ice':0.10,
}

# ═══════════════════════════════════════════════════════════
#  TRACKING TUNING
# ═══════════════════════════════════════════════════════════

DIST_EMA_ALPHA   = 0.15
SPEED_WINDOW     = 15
SPEED_MEDIAN_WIN = 9
SPEED_EMA_ALPHA  = 0.18
MIN_SPEED_KMPH   = 1.5
MAX_SPEED_KMPH   = 180.0
MIN_TRACK_FRAMES = 10
MAX_LOST_FRAMES  = 14
STATIONARY_PX    = 0.8

LK_PARAMS   = dict(winSize=(21,21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, 0.01))
FEAT_PARAMS = dict(maxCorners=10, qualityLevel=0.15, minDistance=8, blockSize=7)

# ═══════════════════════════════════════════════════════════
#  DARK THEME PALETTE
# ═══════════════════════════════════════════════════════════

BG       = '#0f1117'
BG2      = '#1a1d26'
BG3      = '#252836'
BG4      = '#2d3250'
ACCENT   = '#4a9eff'
TEXT     = '#e8e9f0'
TEXT2    = '#8b8fa8'
C_GREEN  = '#22c55e'
C_YELLOW = '#eab308'
C_ORANGE = '#f97316'
C_RED    = '#ef4444'

# ═══════════════════════════════════════════════════════════
#  PHYSICS
# ═══════════════════════════════════════════════════════════

def banked_speed(mu, r, theta_deg):
    g, tan_t = 9.81, math.tan(math.radians(theta_deg))
    denom = 1.0 - mu * tan_t
    if denom <= 0: return float('inf')
    return round(math.sqrt(r * g * (mu + tan_t) / denom) * 3.6, 2)

def vehicle_limit(cls_name, base_speed):
    return base_speed * SAFETY_FACTOR.get(cls_name, 1.0)

def calc_brake_force(cls_name, decel_ms2):
    if decel_ms2 < 0.5: return 0.0
    return round(VEHICLE_MASS.get(cls_name, 1500) * decel_ms2, 0)

# ═══════════════════════════════════════════════════════════
#  COLOR HELPERS
# ═══════════════════════════════════════════════════════════

def spd_color_bgr(s, limit):
    if s >= limit + 15:  return (0,   0, 255)
    if s >= limit:       return (0,  80, 255)
    if s >= limit * 0.9: return (0, 165, 255)
    if s >= limit * 0.7: return (0, 255, 255)
    return (0, 220, 0)

def spd_color_tk(s, limit):
    if limit <= 0: return TEXT2
    if s >= limit + 15:  return C_RED
    if s >= limit:       return C_ORANGE
    if s >= limit * 0.9: return C_YELLOW
    if s >= limit * 0.7: return '#a3e635'
    return C_GREEN

def dist_color_tk(d):
    if d is None: return TEXT2
    if d < 10:    return C_RED
    if d < 25:    return C_ORANGE
    if d < 50:    return C_YELLOW
    return C_GREEN

def dist_dot_char(d):
    if d is None: return '⚪'
    if d < 10:    return '🔴'
    if d < 25:    return '🟠'
    if d < 50:    return '🟡'
    return '🟢'

# ═══════════════════════════════════════════════════════════
#  OPTICAL FLOW HELPERS
# ═══════════════════════════════════════════════════════════

def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua      = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0.0

def focal_distance(box_h, cls_name, focal):
    real_h = REAL_HEIGHTS.get(cls_name, 1.5)
    if box_h < 5: return None
    return round((real_h * focal) / box_h, 2)

def extract_feats(gray, box):
    x1,y1,x2,y2 = (max(0,int(box[0])), max(0,int(box[1])),
                    min(gray.shape[1]-1,int(box[2])),
                    min(gray.shape[0]-1,int(box[3])))
    if x2-x1<10 or y2-y1<10: return None
    roi = gray[y1:y2,x1:x2]
    pts = cv2.goodFeaturesToTrack(roi, mask=None, **FEAT_PARAMS)
    if pts is None or len(pts)==0: return None
    pts[:,0,0]+=x1; pts[:,0,1]+=y1
    return pts

def flow_mag(pg, cg, pts):
    if pts is None or len(pts)==0: return None
    np2,st,_ = cv2.calcOpticalFlowPyrLK(pg,cg,pts,None,**LK_PARAMS)
    if np2 is None or st is None: return None
    if (st.flatten()==1).sum()==0: return None
    go = pts[st.flatten()==1].reshape(-1,2)
    gn = np2[st.flatten()==1].reshape(-1,2)
    if len(go)<3: return None
    d  = gn-go
    return float(np.median(np.hypot(d[:,0],d[:,1])))

# ═══════════════════════════════════════════════════════════
#  WARNING OVERLAY
# ═══════════════════════════════════════════════════════════

def draw_warning_overlay(frame, vehicles, base_safe, flash_state):
    h, w   = frame.shape[:2]
    pw, ph, pad = 270, 142, 10
    x0, y0 = pad, h - ph - pad

    over_speed = [v for v in vehicles if v['show_speed'] and
                  v['speed'] is not None and v['speed'] >= v['limit']]
    with_dist  = [v for v in vehicles if v['distance'] is not None]
    closest    = min(with_dist, key=lambda v: v['distance']) if with_dist else None
    danger     = bool(over_speed) or (closest and closest['distance'] < 10)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0,y0), (x0+pw, y0+ph), (15,18,28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    bcol = (0,0,230) if (danger and flash_state) else (40,44,58)
    cv2.rectangle(frame, (x0,y0), (x0+pw,y0+ph), bcol,
                  3 if danger else 1)
    cv2.rectangle(frame, (x0,y0), (x0+pw,y0+22), (25,30,50), -1)
    cv2.putText(frame, ' WARNING SYSTEM', (x0+4, y0+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (74,158,255), 1, cv2.LINE_AA)

    yt = y0 + 38
    if closest:
        d    = closest['distance']
        dcol = ((0,0,230) if d<10 else (0,115,250) if d<25
                else (0,220,230) if d<50 else (50,200,50))
        dlbl = ('DANGER' if d<10 else 'WARNING' if d<25
                else 'CAUTION' if d<50 else 'SAFE')
        cv2.circle(frame, (x0+16, yt-2), 7, dcol, -1)
        cv2.putText(frame,
                    f'{dlbl}  Closest:{d:.1f}m ({closest["type"]})',
                    (x0+28, yt+2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, dcol, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (x0+16, yt-2), 7, (100,100,100), -1)
        cv2.putText(frame, 'No distance data', (x0+28,yt+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1, cv2.LINE_AA)

    yt += 22
    cv2.line(frame, (x0+6,yt-6), (x0+pw-6,yt-6), (40,44,58), 1)

    cv2.putText(frame, ' OVERSPEED VEHICLES', (x0+4,yt+4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160,160,200), 1, cv2.LINE_AA)
    yt += 18
    if not over_speed:
        cv2.putText(frame, '  All vehicles within limit', (x0+4,yt+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50,200,80), 1, cv2.LINE_AA)
    else:
        for v in over_speed[:3]:
            excess = v['speed'] - v['limit']
            scol   = (0,60,230) if excess>15 else (0,130,255)
            cv2.putText(frame,
                        f'  #{v["id"]} {v["type"]}  {int(v["speed"])}km/h  +{int(excess)} over',
                        (x0+4,yt+4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, scol, 1, cv2.LINE_AA)
            yt += 16
        if len(over_speed) > 3:
            cv2.putText(frame, f'  … and {len(over_speed)-3} more',
                        (x0+4,yt+4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,100,100), 1)

    if danger and flash_state:
        cv2.rectangle(frame, (0,0), (w-1,h-1), (0,0,200), 5)

# ═══════════════════════════════════════════════════════════
#  SPEEDOMETER
# ═══════════════════════════════════════════════════════════

def draw_speedometer(frame, speed, limit, cx, cy, r=34):
    col = spd_color_bgr(speed, limit)
    cv2.circle(frame,(cx,cy),r,(20,20,20),-1)
    cv2.circle(frame,(cx,cy),r,col,2)
    sa = min(limit/MAX_SPEED_KMPH,1.0)*270-135
    sx = int(cx+(r-4)*math.cos(math.radians(sa)))
    sy = int(cy+(r-4)*math.sin(math.radians(sa)))
    cv2.circle(frame,(sx,sy),3,(0,200,0),-1)
    na = min(speed/MAX_SPEED_KMPH,1.0)*270-135
    ex = int(cx+(r-9)*math.cos(math.radians(na)))
    ey = int(cy+(r-9)*math.sin(math.radians(na)))
    cv2.line(frame,(cx,cy),(ex,ey),col,2)
    t = str(int(speed))
    (tw,th),_ = cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,0.40,2)
    cv2.putText(frame,t,(cx-tw//2,cy+th//2),cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),2)

# ═══════════════════════════════════════════════════════════
#  TRACK + TRACKER
# ═══════════════════════════════════════════════════════════

class Track:
    def __init__(self,tid,cls_name,box):
        self.id=tid; self.cls_name=cls_name; self.box=box
        self.lost=0; self.frame_count=1; self.flow_pts=None
        self._dist_ema=None; self.time_alive=0.0
        self._dist_hist=deque(maxlen=SPEED_WINDOW)
        self._raw_speeds=deque(maxlen=SPEED_MEDIAN_WIN)
        self.smooth_speed=0.0; self.is_stationary=False
        self._prev_speed=None; self.acceleration=0.0; self.brake_force_n=0.0

    def push_distance(self,d):
        if d is None: return self._dist_ema
        self._dist_ema = (d if self._dist_ema is None
                          else DIST_EMA_ALPHA*d+(1-DIST_EMA_ALPHA)*self._dist_ema)
        return self._dist_ema

    def push_speed(self,vdt):
        if self._dist_ema is None: return
        self.time_alive+=vdt
        self._dist_hist.append((self.time_alive,self._dist_ema))
        if len(self._dist_hist)>=5:
            t0,d0=self._dist_hist[0]; t1,d1=self._dist_hist[-1]
            dt=t1-t0
            if dt>0:
                sp_kh=min(abs(d1-d0)/dt*3.6,MAX_SPEED_KMPH)
                if not (self._raw_speeds and
                        sp_kh>max(self._raw_speeds)*3 and
                        len(self._raw_speeds)>=3):
                    self._raw_speeds.append(sp_kh)
                med=float(np.median(self._raw_speeds)) if self._raw_speeds else 0.0
                if med<MIN_SPEED_KMPH: med=0.0
                self.smooth_speed=(med if self.smooth_speed==0.0
                    else SPEED_EMA_ALPHA*med+(1-SPEED_EMA_ALPHA)*self.smooth_speed)
        if self._prev_speed is not None and vdt>0:
            dv=(self.smooth_speed-self._prev_speed)/3.6
            self.acceleration=dv/vdt
            self.brake_force_n=(calc_brake_force(self.cls_name,-self.acceleration)
                                if self.acceleration<-0.5 else 0.0)
        self._prev_speed=self.smooth_speed

    @property
    def display_dist(self): return self._dist_ema


class Tracker:
    def __init__(self): self.tracks={}; self.next_id=0

    def update(self,detections):
        mt=set(); md=set(); pairs=[]
        for tid,trk in self.tracks.items():
            for di,(db,_) in enumerate(detections):
                s=iou(trk.box,db)
                if s>0.15: pairs.append((s,tid,di))
        pairs.sort(reverse=True)
        for s,tid,di in pairs:
            if tid in mt or di in md: continue
            db,dc=detections[di]
            t=self.tracks[tid]; t.box=db; t.cls_name=dc
            t.lost=0; t.frame_count+=1; mt.add(tid); md.add(di)
        for di,(db,dc) in enumerate(detections):
            if di not in md:
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]=Track(tid,dc,db)
        for tid in list(self.tracks):
            if tid not in mt:
                self.tracks[tid].lost+=1
                if self.tracks[tid].lost>MAX_LOST_FRAMES:
                    del self.tracks[tid]
        return list(self.tracks.values())

# ═══════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self._lock=threading.Lock()
        self.frame=None; self.vehicle_data=[]
        self.confidence=0.40; self.friction=0.70
        self.radius=30.0; self.bank_angle=10.0
        self.base_safe=banked_speed(0.70,30.0,10.0)
        # Per-vehicle safe speed dict  {cls_name: km/h}
        self.vehicle_limits={cls: vehicle_limit(cls, self.base_safe)
                             for cls in CLASS_NAMES}
        self.overspeed=False; self.fps=0.0
        self.frame_count=0; self.total_frames=0
        self.max_speed_seen=0.0; self.max_speed_cls=''
        self.speed_alerts=0; self.running=True; self.paused=False
        self._src_lock=threading.Lock(); self._new_source=None

    def push(self,frame,vehicles,overspeed,fps,fc,mx_spd,mx_cls,alerts):
        with self._lock:
            self.frame=frame.copy() if frame is not None else None
            self.vehicle_data=vehicles; self.overspeed=overspeed
            self.fps=fps; self.frame_count=fc
            self.max_speed_seen=mx_spd; self.max_speed_cls=mx_cls
            self.speed_alerts=alerts

    def set_conf(self,v):
        with self._lock: self.confidence=v

    def set_curve(self,mu,r,b):
        with self._lock:
            self.friction=mu; self.radius=r; self.bank_angle=b
            self.base_safe=banked_speed(mu,r,b)
            self.vehicle_limits={cls: vehicle_limit(cls, self.base_safe)
                                  for cls in CLASS_NAMES}

    def request_source(self,source):
        with self._src_lock: self._new_source=source

    def poll_source(self):
        with self._src_lock:
            s=self._new_source; self._new_source=None; return s

    def get_frame(self):
        with self._lock:
            return self.frame.copy() if self.frame is not None else None

    def get_vehicles(self):
        with self._lock: return list(self.vehicle_data)

    def get_curve_params(self):
        with self._lock:
            return (self.confidence,self.friction,self.radius,
                    self.bank_angle,self.base_safe)

    def get_vehicle_limits(self):
        with self._lock: return dict(self.vehicle_limits)

# ═══════════════════════════════════════════════════════════
#  VIDEO WORKER THREAD (UDP MODIFIED)
# ═══════════════════════════════════════════════════════════

def video_worker(state,model,initial_source):
    tracker=Tracker(); prev_gray=None
    fps_disp=30.0; t_prev=time.time()
    frame_cnt=0; max_spd=0.0; max_cls=''; alerts=0
    device='cuda' if torch.cuda.is_available() else 'cpu'
    flash_tick=0; flash_state=False

    def open_source(src):
        nonlocal prev_gray,frame_cnt,tracker
        try:
            cam_idx=int(src); is_cam=True
            cap=cv2.VideoCapture(cam_idx)
        except (ValueError,TypeError):
            is_cam=False; cap=cv2.VideoCapture(str(src))
        if not cap.isOpened():
            cap.release(); return None,False,0,1/30.0
        ok,_=cap.read()
        if not ok:
            cap.release(); return None,False,0,1/30.0
        if not is_cam: cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
        vdt=1.0/max(cap.get(cv2.CAP_PROP_FPS) or 30.0,1)
        prev_gray=None; frame_cnt=0; tracker=Tracker()
        return cap,is_cam,total,vdt

    cap,is_camera,total,vdt=open_source(initial_source)

    while state.running:
        new_src=state.poll_source()
        if new_src is not None:
            if cap: cap.release(); cap=None
            cap,is_camera,total,vdt=open_source(new_src)

        if cap is None or state.paused:
            time.sleep(0.05); continue

        ret,frame=cap.read()
        if not ret:
            if is_camera: time.sleep(0.01)
            else: cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            continue

        frame_cnt+=1
        flash_tick=(flash_tick+1)%20
        if flash_tick==0: flash_state=not flash_state

        conf,mu,radius,bank,base_safe=state.get_curve_params()
        curr_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        res=model(frame,conf=conf,iou=0.45,imgsz=640,device=device,verbose=False)[0]
        detections=[]
        for b in res.boxes:
            cid=int(b.cls[0])
            cname=CLASS_NAMES[cid] if cid<len(CLASS_NAMES) else 'unknown'
            x1,y1,x2,y2=map(int,b.xyxy[0])
            detections.append(((x1,y1,x2,y2),cname))

        tracks=tracker.update(detections)
        overspeed=False; vehicles=[]
        W=frame.shape[1]; focal=FOCAL_LENGTH or (W*0.85)

        for trk in tracks:
            x1,y1,x2,y2=trk.box; box_h=y2-y1
            trk.push_distance(focal_distance(box_h,trk.cls_name,focal))
            trk.push_speed(vdt)
            if prev_gray is not None:
                if trk.flow_pts is None:
                    trk.flow_pts=extract_feats(prev_gray,trk.box)
                mag=flow_mag(prev_gray,curr_gray,trk.flow_pts)
                trk.is_stationary=mag is not None and mag<STATIONARY_PX
                if frame_cnt%15==0:
                    trk.flow_pts=extract_feats(curr_gray,trk.box)
            if trk.is_stationary:
                trk.smooth_speed=max(trk.smooth_speed*0.7,0.0)

            veh_lim=vehicle_limit(trk.cls_name,base_safe)
            show_s=trk.frame_count>=MIN_TRACK_FRAMES
            if show_s and trk.smooth_speed>veh_lim:
                overspeed=True; alerts+=1
            if trk.smooth_speed>max_spd:
                max_spd=trk.smooth_speed; max_cls=trk.cls_name

            color=CLASS_COLORS_BGR.get(trk.cls_name,(200,200,200))
            scol=spd_color_bgr(trk.smooth_speed,veh_lim)
            bcol=(0,0,255) if (show_s and trk.smooth_speed>=veh_lim) else color
            bthk=3         if (show_s and trk.smooth_speed>=veh_lim) else 2
            cv2.rectangle(frame,(x1,y1),(x2,y2),bcol,bthk)

            dist_txt=f'{trk.display_dist:.1f}m' if trk.display_dist else '?m'
            lbl=f'{trk.cls_name} | {dist_txt}'
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.50,2)
            cv2.rectangle(frame,(x1,y1-th-12),(x1+tw+6,y1),color,-1)
            cv2.putText(frame,lbl,(x1+3,y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.50,(255,255,255),2,cv2.LINE_AA)

            if show_s:
                stx=('stationary' if trk.is_stationary and
                     trk.smooth_speed<MIN_SPEED_KMPH
                     else f'{int(trk.smooth_speed)} km/h')
                (sw,sh),_=cv2.getTextSize(stx,cv2.FONT_HERSHEY_SIMPLEX,0.58,2)
                cv2.rectangle(frame,(x1,y2),(x1+sw+6,y2+sh+10),(0,0,0),-1)
                cv2.putText(frame,stx,(x1+3,y2+sh+3),
                            cv2.FONT_HERSHEY_SIMPLEX,0.58,scol,2,cv2.LINE_AA)
                cv2.putText(frame,f'Lim:{veh_lim:.0f}',(x1+3,y2+sh+20),
                            cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,200,0),1,cv2.LINE_AA)
                if trk.brake_force_n>1000:
                    cv2.putText(frame,f'F={trk.brake_force_n/1000:.1f}kN',
                                (x1+3,y2+sh+34),
                                cv2.FONT_HERSHEY_SIMPLEX,0.36,(0,100,255),1,cv2.LINE_AA)
                draw_speedometer(frame,trk.smooth_speed,veh_lim,
                                 (x1+x2)//2,max(y1-48,42),r=33)

            vehicles.append({
                'id':trk.id,'type':trk.cls_name,
                'speed':trk.smooth_speed if show_s else None,
                'distance':trk.display_dist,'limit':veh_lim,
                'brake_kn':trk.brake_force_n/1000 if trk.brake_force_n>0 else 0.0,
                'stationary':trk.is_stationary,'show_speed':show_s,
            })

        draw_warning_overlay(frame,vehicles,base_safe,flash_state)

        if overspeed:
            over_types=list({v['type'] for v in vehicles
                             if v['show_speed'] and v['speed'] and v['speed']>=v['limit']})
            txt=f'  \u26a0 OVERSPEED: {", ".join(over_types[:3])}  '
            (bw,bh),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.80,2)
            bx=(frame.shape[1]-bw)//2
            cv2.rectangle(frame,(bx-10,8),(bx+bw+10,52),(0,0,200),-1)
            cv2.putText(frame,txt,(bx,42),
                        cv2.FONT_HERSHEY_SIMPLEX,0.80,(255,255,255),2,cv2.LINE_AA)

        t_now=time.time()
        fps_disp=0.9*fps_disp+0.1*(1.0/max(t_now-t_prev,1e-4))
        t_prev=t_now
        cv2.putText(frame,f'FPS:{fps_disp:.1f}',(frame.shape[1]-110,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,255,255),2)
        if total>0:
            pw=int(frame.shape[1]*frame_cnt/total)
            cv2.rectangle(frame,(0,frame.shape[0]-4),(pw,frame.shape[0]),(0,180,0),-1)

        state.push(frame,vehicles,overspeed,fps_disp,frame_cnt,max_spd,max_cls,alerts)
        state.total_frames=total
        prev_gray=curr_gray

        # ═══════════════════════════════════════════════════════════
        #  SEND UDP JSON DATA TO ESP32 NODE
        # ═══════════════════════════════════════════════════════════
        
        # 1. Isolate the absolute closest distance
        valid_distances = [v['distance'] for v in vehicles if v['distance'] is not None]
        closest_dist = min(valid_distances) if valid_distances else 999.0

        # 2. Extract the worst over-speeding vehicle (for max speed/type mapping)
        over_v = [v for v in vehicles if v['show_speed'] and v['speed'] and v['speed'] >= v['limit']]
        worst_v = max(over_v, key=lambda x: x['speed'] - x['limit']) if over_v else None

        # 3. Serialize and push the payload seamlessly
        payload = {
            "dist": round(closest_dist, 1),
            "overspeed": bool(worst_v),
            "limit": int(worst_v['limit']) if worst_v else 0,
            "type": worst_v['type'] if worst_v else ""
        }
        
        try:
            udp_socket.sendto(json.dumps(payload).encode('utf-8'), (ESP32_IP, ESP32_PORT))
        except Exception:
            pass # Suppress socket errors to prevent UI blockage

    if cap: cap.release()

# ═══════════════════════════════════════════════════════════
#  TKINTER APPLICATION
# ═══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self,state,initial_source):
        super().__init__()
        self.state=state; self.initial_source=initial_source
        self.title('Vehicle Detection System  —  Curve Physics  v9')
        self.configure(bg=BG)
        self.geometry('1440x820'); self.minsize(1200,700)
        self._last_vehicle_fp=None
        self._setup_styles()
        self._build_left_panel()
        self._build_notebook()
        self.protocol('WM_DELETE_WINDOW',self._on_close)
        self._tick()

    def _setup_styles(self):
        s=ttk.Style(self); s.theme_use('clam')
        s.configure('.',background=BG,foreground=TEXT,font=('Segoe UI',10))
        s.configure('TNotebook',background=BG2,borderwidth=0)
        s.configure('TNotebook.Tab',background=BG3,foreground=TEXT2,
                    padding=[16,7],font=('Segoe UI',10,'bold'))
        s.map('TNotebook.Tab',background=[('selected',BG)],
              foreground=[('selected',ACCENT)])
        s.configure('TFrame',background=BG)
        s.configure('TLabel',background=BG,foreground=TEXT)
        s.configure('TScale',background=BG2,troughcolor=BG3,sliderthickness=18)
        s.configure('Treeview',background=BG3,foreground=TEXT,
                    fieldbackground=BG3,rowheight=28,font=('Segoe UI',9))
        s.configure('Treeview.Heading',background=BG2,foreground=ACCENT,
                    font=('Segoe UI',9,'bold'),relief='flat')
        s.map('Treeview',background=[('selected',BG4)],
              foreground=[('selected',TEXT)])
        s.configure('TButton',background=BG3,foreground=TEXT,
                    font=('Segoe UI',10),padding=[10,6],relief='flat')
        s.map('TButton',background=[('active',BG4)])
        s.configure('TCombobox',background=BG3,foreground=TEXT,
                    fieldbackground=BG3,selectbackground=BG4,
                    selectforeground=TEXT)
        s.configure('Accent.TButton',background=ACCENT,foreground='#000',
                    font=('Segoe UI',10,'bold'),padding=[10,6])
        s.map('Accent.TButton',background=[('active','#3a8eef')])

    def _build_left_panel(self):
        lp=tk.Frame(self,bg=BG2,width=230)
        lp.pack(side='left',fill='y',padx=(8,4),pady=8)
        lp.pack_propagate(False)

        tk.Label(lp,text='🚦',font=('Segoe UI',28),bg=BG2).pack(pady=(14,2))
        tk.Label(lp,text='VEHICLE\nDETECTION',font=('Segoe UI',12,'bold'),
                 fg=ACCENT,bg=BG2,justify='center').pack()
        self._hline(lp)

        tk.Label(lp,text='VIDEO SOURCE',font=('Segoe UI',9,'bold'),
                 fg=TEXT2,bg=BG2).pack(pady=(4,4))

        self.src_type=tk.StringVar(value='file')
        rf=tk.Frame(lp,bg=BG2); rf.pack(fill='x',padx=10,pady=(0,6))
        for val,txt in [('camera','Camera'),('file','Video File')]:
            tk.Radiobutton(rf,text=txt,variable=self.src_type,value=val,
                           bg=BG2,fg=TEXT,selectcolor=BG3,
                           activebackground=BG2,activeforeground=ACCENT,
                           command=self._on_src_type).pack(side='left',padx=4)

        self._src_container=tk.Frame(lp,bg=BG2,height=82)
        self._src_container.pack(fill='x',padx=10,pady=2)
        self._src_container.pack_propagate(False)
        self._src_container.grid_columnconfigure(0,weight=1)

        self.cam_frame=tk.Frame(self._src_container,bg=BG2)
        self.cam_frame.grid(row=0,column=0,sticky='nsew')
        tk.Label(self.cam_frame,text='Camera index:',
                 font=('Segoe UI',9),fg=TEXT2,bg=BG2).pack(side='left',pady=6)
        self.cam_idx_var=tk.StringVar(value='0')
        ttk.Combobox(self.cam_frame,textvariable=self.cam_idx_var,
                     values=['0','1','2','3'],state='readonly',
                     width=4).pack(side='left',padx=6)

        self.file_frame=tk.Frame(self._src_container,bg=BG2)
        self.file_frame.grid(row=0,column=0,sticky='nsew')

        self._full_file_path=(str(self.initial_source)
                              if isinstance(self.initial_source,str) else '')
        self.file_path_var=tk.StringVar(
            value=os.path.basename(self._full_file_path)
                  if self._full_file_path else '')

        tk.Label(self.file_frame,textvariable=self.file_path_var,
                 font=('Segoe UI',8),fg=TEXT2,bg=BG2,
                 wraplength=190,justify='left',anchor='w').pack(fill='x')
        ttk.Button(self.file_frame,text='📂  Browse…',
                   command=self._browse_file).pack(fill='x',pady=(4,0))

        self._on_src_type()

        ttk.Button(lp,text='▶  Connect Source',style='Accent.TButton',
                   command=self._connect_source).pack(padx=14,pady=8,fill='x')

        self._hline(lp)

        tk.Label(lp,text='Detection Confidence',font=('Segoe UI',9,'bold'),
                 fg=TEXT2,bg=BG2).pack(pady=(4,2))
        self.conf_var=tk.DoubleVar(value=0.40)
        self.conf_lbl=tk.Label(lp,text='0.40',font=('Segoe UI',18,'bold'),
                                fg=ACCENT,bg=BG2)
        self.conf_lbl.pack()
        ttk.Scale(lp,from_=0.10,to=0.90,variable=self.conf_var,
                  orient='horizontal',length=190,
                  command=self._on_conf).pack(padx=10,pady=4)
        tk.Label(lp,text='0.10 ←          → 0.90',
                 font=('Segoe UI',8),fg=TEXT2,bg=BG2).pack()
        self._hline(lp)

        tk.Label(lp,text='LIVE STATS',font=('Segoe UI',9,'bold'),
                 fg=TEXT2,bg=BG2).pack(pady=(4,6))
        self.stats={}
        for key,lbl,default,col in [
            ('vehicles','Vehicles','0',TEXT),
            ('fps','FPS','0.0',ACCENT),
            ('max_speed','Max Speed','0 km/h',TEXT),
            ('max_cls','Vehicle','—',TEXT2),
            ('alerts','Alerts','0',TEXT),
        ]:
            row=tk.Frame(lp,bg=BG2); row.pack(fill='x',padx=10,pady=2)
            tk.Label(row,text=lbl,font=('Segoe UI',9),fg=TEXT2,
                     bg=BG2,anchor='w',width=10).pack(side='left')
            w=tk.Label(row,text=default,font=('Segoe UI',9,'bold'),
                       fg=col,bg=BG2,anchor='e')
            w.pack(side='right'); self.stats[key]=w
        self._hline(lp)

        self.pause_btn=ttk.Button(lp,text='⏸  Pause',command=self._toggle_pause)
        self.pause_btn.pack(padx=14,pady=4,fill='x')
        ttk.Button(lp,text='✕  Quit',command=self._on_close
                   ).pack(padx=14,pady=4,fill='x')
        self._hline(lp)

        self.warn_pill=tk.Label(lp,text='✔  All Safe',
                                 font=('Segoe UI',10,'bold'),
                                 bg=BG2,fg=C_GREEN,
                                 wraplength=200,justify='center')
        self.warn_pill.pack(pady=10,padx=6)

    def _on_src_type(self):
        if self.src_type.get()=='camera':
            self.cam_frame.tkraise()
        else:
            self.file_frame.tkraise()

    def _browse_file(self):
        path=filedialog.askopenfilename(
            title='Select Video File',
            filetypes=[('Video files','*.mp4 *.avi *.mkv *.mov *.m4v *.wmv *.webm'),
                       ('All files','*.*')])
        if path:
            self._full_file_path=path
            self.file_path_var.set(os.path.basename(path))

    def _connect_source(self):
        if self.src_type.get()=='camera':
            src=self.cam_idx_var.get()
        else:
            src=self._full_file_path
            if not src:
                messagebox.showwarning('No file',
                    'Please browse and select a video file first.')
                return
        self._last_vehicle_fp=None
        self.state.request_source(src)

    def _build_notebook(self):
        self.nb=ttk.Notebook(self)
        self.nb.pack(side='left',fill='both',expand=True,padx=(0,8),pady=8)
        self.tab1=ttk.Frame(self.nb)
        self.nb.add(self.tab1,text='  📊  Vehicle Data  ')
        self._build_tab1()
        self.tab2=ttk.Frame(self.nb)
        self.nb.add(self.tab2,text='  ⚙  Curve Settings  ')
        self._build_tab2()

    def _build_tab1(self):
        self.warn_banner=tk.Label(self.tab1,text='',
                                   font=('Segoe UI',10,'bold'),
                                   bg=BG,fg=C_RED,pady=5)
        self.warn_banner.pack(side='bottom',fill='x',padx=6,pady=(0,4))

        main=tk.Frame(self.tab1,bg=BG)
        main.pack(fill='both',expand=True,padx=6,pady=6)

        right=tk.Frame(main,bg=BG2)
        right.pack(side='right',fill='y',padx=(6,0))
        tk.Label(right,text='  VEHICLE DATA',font=('Segoe UI',11,'bold'),
                 fg=ACCENT,bg=BG2,anchor='w').pack(fill='x',padx=6,pady=(8,4))

        leg=tk.Frame(right,bg=BG3); leg.pack(fill='x',padx=6,pady=(0,4))
        for col,txt in [(C_GREEN,'>50m'),(C_YELLOW,'25-50m'),
                        (C_ORANGE,'10-25m'),(C_RED,'<10m')]:
            tk.Label(leg,text='●',fg=col,bg=BG3,
                     font=('Segoe UI',11)).pack(side='left',padx=(5,1))
            tk.Label(leg,text=txt,fg=TEXT2,bg=BG3,
                     font=('Segoe UI',8)).pack(side='left',padx=(0,4))

        cols=('ID','Type','Speed','Dist','Limit','Brake')
        self.tree=ttk.Treeview(right,columns=cols,show='headings',height=18)
        widths={'ID':44,'Type':88,'Speed':72,'Dist':64,'Limit':54,'Brake':58}
        for c in cols:
            self.tree.heading(c,text=c)
            self.tree.column(c,width=widths[c],anchor='center',minwidth=widths[c])
        vsb=ttk.Scrollbar(right,orient='vertical',command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side='left',fill='both',expand=True,padx=(6,0),pady=(0,6))
        vsb.pack(side='left',fill='y',pady=(0,6),padx=(0,4))

        vid_card=tk.Frame(main,bg=BG2)
        vid_card.pack(side='left',fill='both',expand=True)
        hdr=tk.Frame(vid_card,bg=BG2); hdr.pack(fill='x',padx=6,pady=(8,2))
        tk.Label(hdr,text='  LIVE FEED',font=('Segoe UI',11,'bold'),
                 fg=ACCENT,bg=BG2,anchor='w').pack(side='left')
        self.src_indicator=tk.Label(hdr,text='●  connecting…',
                                     font=('Segoe UI',9),fg=C_YELLOW,bg=BG2)
        self.src_indicator.pack(side='right',padx=8)
        self.video_canvas=tk.Canvas(vid_card,bg='#000',
                                     highlightthickness=1,
                                     highlightbackground=BG3)
        self.video_canvas.pack(fill='both',expand=True,padx=6,pady=6)
        self._vimg=None

    def _build_tab2(self):
        outer=tk.Frame(self.tab2,bg=BG)
        outer.pack(fill='both',expand=True,padx=24,pady=16)
        tk.Label(outer,text='Mountain Curve Physics Settings',
                 font=('Segoe UI',15,'bold'),fg=ACCENT,bg=BG).pack(pady=(0,18))

        row=tk.Frame(outer,bg=BG); row.pack(fill='x',pady=6)
        tk.Label(row,text='Road Surface:',font=('Segoe UI',10),
                 fg=TEXT2,bg=BG,width=26,anchor='w').pack(side='left')
        self.surface_var=tk.StringVar(value='Dry Asphalt')
        cb=ttk.Combobox(row,textvariable=self.surface_var,
                        values=list(ROAD_CONDITIONS.keys()),
                        state='readonly',width=18)
        cb.pack(side='left',padx=8)
        cb.bind('<<ComboboxSelected>>',self._on_surface)

        self._add_slider(outer,'Friction Coefficient (μ)','mu_var','mu_lbl',0.05,0.90,0.70,2)
        self._add_slider(outer,'Curve Radius (m)','rad_var','rad_lbl',10.0,200.0,30.0,1)
        self._add_slider(outer,'Bank Angle (°)','bank_var','bank_lbl',0.0,30.0,10.0,1)
        self._hline(outer,bg=BG)

        tk.Label(outer,
                 text='MAX SAFE SPEED PER VEHICLE TYPE',
                 font=('Segoe UI',10,'bold'),fg=TEXT2,bg=BG).pack(pady=(0,2))
        tk.Label(outer,
                 text='v = √[ r · g · (μ + tan θ) / (1 − μ · tan θ) ]  ×  safety factor',
                 font=('Consolas',9),fg=TEXT2,bg=BG).pack(pady=(0,8))

        grid_frame=tk.Frame(outer,bg=BG)
        grid_frame.pack(fill='x',pady=(0,4))
        COLS=3
        for i in range(COLS):
            grid_frame.columnconfigure(i,weight=1)

        self._mss_labels={} 

        for idx,cls in enumerate(CLASS_NAMES):
            col=idx % COLS
            row_i=idx // COLS

            card=tk.Frame(grid_frame,bg=BG3,relief='flat',bd=0)
            card.grid(row=row_i,column=col,padx=4,pady=4,sticky='ew')

            tk.Label(card,text=cls,font=('Segoe UI',8,'bold'),
                     fg=TEXT2,bg=BG3,anchor='w').pack(
                         fill='x',padx=8,pady=(6,0))

            sf=SAFETY_FACTOR.get(cls,1.0)
            tk.Label(card,text=f'×{sf:.2f} safety',
                     font=('Segoe UI',7),fg=BG4,bg=BG3,anchor='w').pack(
                         fill='x',padx=8)

            spd_lbl=tk.Label(card,text='— km/h',
                             font=('Segoe UI',14,'bold'),
                             fg=C_GREEN,bg=BG3,anchor='w')
            spd_lbl.pack(fill='x',padx=8,pady=(2,6))

            bar=tk.Canvas(card,height=4,bg=BG4,
                          highlightthickness=0,bd=0)
            bar.pack(fill='x',padx=8,pady=(0,6))

            self._mss_labels[cls]=(spd_lbl,bar)

        self._hline(outer,bg=BG)

        tk.Label(outer,text='Vehicle Properties Reference',
                 font=('Segoe UI',10,'bold'),fg=ACCENT,bg=BG).pack(pady=(4,4))
        cols2=('Vehicle','Safety ×','Safe Speed (km/h)','Mass (kg)')
        self.lim_tree=ttk.Treeview(outer,columns=cols2,show='headings',height=9)
        widths2={'Vehicle':130,'Safety ×':80,'Safe Speed (km/h)':160,'Mass (kg)':100}
        for c in cols2:
            self.lim_tree.heading(c,text=c)
            self.lim_tree.column(c,width=widths2[c],anchor='center')
        self.lim_tree.pack(fill='x',pady=(0,6))

        self._refresh_limits()
        self._refresh_mss()

    def _add_slider(self,parent,label,var_name,lbl_name,lo,hi,init,dec):
        row=tk.Frame(parent,bg=BG); row.pack(fill='x',pady=8)
        tk.Label(row,text=label,font=('Segoe UI',10),fg=TEXT2,
                 bg=BG,width=28,anchor='w').pack(side='left')
        var=tk.DoubleVar(value=init); setattr(self,var_name,var)
        val_lbl=tk.Label(row,text=f'{init:.{dec}f}',
                          font=('Segoe UI',12,'bold'),fg=ACCENT,
                          bg=BG,width=7,anchor='e')
        val_lbl.pack(side='right',padx=8); setattr(self,lbl_name,val_lbl)
        fmt=f'{{:.{dec}f}}'
        def on_slide(v,L=val_lbl,f=fmt):
            L.config(text=f.format(float(v))); self._on_curve()
        ttk.Scale(row,from_=lo,to=hi,variable=var,
                  orient='horizontal',length=280,
                  command=on_slide).pack(side='left',padx=8)

    def _on_conf(self,v):
        val=round(float(v),2)
        self.conf_lbl.config(text=f'{val:.2f}')
        self.state.set_conf(val)

    def _on_surface(self,_=None):
        mu=ROAD_CONDITIONS.get(self.surface_var.get(),0.70)
        self.mu_var.set(mu); self.mu_lbl.config(text=f'{mu:.2f}')
        self._on_curve()

    def _on_curve(self):
        self.state.set_curve(self.mu_var.get(),self.rad_var.get(),self.bank_var.get())
        self._refresh_mss()
        self._refresh_limits()

    def _toggle_pause(self):
        self.state.paused=not self.state.paused
        self.pause_btn.config(text='▶  Resume' if self.state.paused else '⏸  Pause')

    def _on_close(self):
        self.state.running=False; self.destroy()

    def _refresh_mss(self):
        if not hasattr(self,'mu_var'): return
        base=banked_speed(self.mu_var.get(),self.rad_var.get(),self.bank_var.get())
        for cls,(spd_lbl,bar) in self._mss_labels.items():
            limit=vehicle_limit(cls,base)
            if limit==float('inf'):
                txt='∞ km/h'; fg=C_GREEN
            else:
                txt=f'{limit:.1f} km/h'
                fg=(C_GREEN if limit>50 else
                    C_YELLOW if limit>30 else C_RED)
            spd_lbl.config(text=txt,fg=fg)

            self.after(10, lambda b=bar, lim=limit, c=fg: self._draw_bar(b,lim,c))

    def _draw_bar(self,bar,limit,color):
        bar.update_idletasks()
        w=bar.winfo_width()
        if w<4: return
        bar.delete('all')
        bar.create_rectangle(0,0,w,4,fill=BG4,outline='')
        frac=min(limit/MAX_SPEED_KMPH,1.0) if limit!=float('inf') else 1.0
        filled=max(4,int(w*frac))
        bar.create_rectangle(0,0,filled,4,fill=color,outline='')

    def _refresh_limits(self):
        if not hasattr(self,'mu_var'): return
        base=banked_speed(self.mu_var.get(),self.rad_var.get(),self.bank_var.get())
        for r in self.lim_tree.get_children(): self.lim_tree.delete(r)
        for cls in CLASS_NAMES:
            lim=vehicle_limit(cls,base)
            lim_txt=f'{lim:.1f}' if lim!=float('inf') else '∞'
            tag=('danger' if lim<30 else ('warn' if lim<50 else 'safe'))
            self.lim_tree.insert('','end',
                values=(cls,f'×{SAFETY_FACTOR.get(cls,1.0):.2f}',
                        lim_txt,f'{VEHICLE_MASS.get(cls,0):,}'),
                tags=(tag,))
        self.lim_tree.tag_configure('safe',  foreground=C_GREEN)
        self.lim_tree.tag_configure('warn',  foreground=C_YELLOW)
        self.lim_tree.tag_configure('danger',foreground=C_RED)

    def _tick(self):
        self._refresh_video()
        self._refresh_table()
        self._refresh_stats()
        self.after(40,self._tick)

    def _refresh_video(self):
        frame=self.state.get_frame()
        if frame is None:
            self.src_indicator.config(text='●  waiting…',fg=C_YELLOW); return
        self.src_indicator.config(text='●  live',fg=C_GREEN)
        cw=max(self.video_canvas.winfo_width(),200)
        ch=max(self.video_canvas.winfo_height(),150)
        if cw<50 or ch<50: return
        h,w=frame.shape[:2]
        if w<=0 or h<=0: return
        scale=min(cw/w,ch/h)
        nw,nh=max(1,int(w*scale)),max(1,int(h*scale))
        resized=cv2.resize(frame,(nw,nh))
        img=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)))
        self.video_canvas.config(width=nw,height=nh)
        self.video_canvas.create_image(0,0,anchor='nw',image=img)
        self._vimg=img

    def _refresh_table(self):
        vehicles=self.state.get_vehicles()
        def fp(v):
            spd=int(v['speed']) if v['speed'] is not None and v['show_speed'] else -1
            return (v['id'],spd,round(v['distance'] or 999,0))
        cur_fp=frozenset(fp(v) for v in vehicles)
        if cur_fp==self._last_vehicle_fp: return
        self._last_vehicle_fp=cur_fp
        for r in self.tree.get_children(): self.tree.delete(r)
        for v in sorted(vehicles,key=lambda x:x['distance'] or 999):
            spd=v['speed']; dist=v['distance']; lim=v['limit']
            brk=v['brake_kn']; show=v['show_speed']
            spd_txt=f"{int(spd)} km/h" if spd is not None and show else 'tracking…'
            dist_txt=f"{dist:.1f} m" if dist else '?'
            brk_txt=f"{brk:.1f} kN" if brk>0 else '—'
            tag=('over' if spd and show and spd>=lim else
                 'caution' if spd and show and spd>=lim*.9 else 'ok')
            self.tree.insert('','end',
                values=(f'{dist_dot_char(dist)}#{v["id"]}',v['type'],
                        spd_txt,dist_txt,f'{lim:.0f}',brk_txt),
                tags=(tag,))
        self.tree.tag_configure('over',   foreground=C_RED)
        self.tree.tag_configure('caution',foreground=C_YELLOW)
        self.tree.tag_configure('ok',     foreground=TEXT)

    def _refresh_stats(self):
        vehicles=self.state.get_vehicles()
        max_cls=self.state.max_speed_cls
        max_lim=vehicle_limit(max_cls, self.state.base_safe) if max_cls else 99
        self.stats['vehicles'].config(text=str(len(vehicles)))
        self.stats['fps'].config(text=f'{self.state.fps:.1f}')
        self.stats['max_speed'].config(
            text=f'{int(self.state.max_speed_seen)} km/h',
            fg=spd_color_tk(self.state.max_speed_seen, max_lim))
        self.stats['max_cls'].config(text=max_cls or '—')
        self.stats['alerts'].config(
            text=str(self.state.speed_alerts),
            fg=C_RED if self.state.speed_alerts>0 else C_GREEN)
        if self.state.overspeed:
            self.warn_pill.config(text='⚠  OVERSPEED\nDETECTED',fg=C_RED)
            over_types=list({v['type'] for v in vehicles
                             if v['show_speed'] and v['speed']
                             and v['speed']>=v['limit']})
            detail=(', '.join(over_types[:4]) if over_types
                    else 'vehicle(s) exceed their limit')
            self.warn_banner.config(
                text=f'⚠  OVERSPEED: {detail}  —  limits vary by vehicle type',
                fg=C_RED,bg='#2d0000')
        else:
            self.warn_pill.config(text='✔  All Safe',fg=C_GREEN)
            self.warn_banner.config(
                text='✔  All vehicles within their per-type safe curve speed limits',
                fg=C_GREEN,bg=BG)

    def _hline(self, parent, bg=BG2):
        tk.Frame(parent, bg=BG3, height=1).pack(fill='x', pady=8, padx=8)

# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    print('\n'+'='*70)
    print('  Vehicle Detection System  —  GUI Edition  v9  (per-vehicle limits)')
    print('='*70)
    model=YOLO(DEFAULT_MODEL)
    print('✅  Model loaded')
    state=SharedState()
    t=threading.Thread(target=video_worker,
                       args=(state,model,DEFAULT_VIDEO),daemon=True)
    t.start()
    app=App(state,DEFAULT_VIDEO)
    app.mainloop()
    state.running=False
    udp_socket.close()
    print('Done.')

if __name__=='__main__':
    main()