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

DEFAULT_MODEL = r'C:\Final project\runs\vehicle_detection_v1\weights\best v2.pt'
DEFAULT_VIDEO_1 = r'C:\Users\Nadhu\Downloads\stock-footage-uttarakhand-in-may-the-traffic-on-a-curvy-road-surrounded-by-mountains-under-blue-sky.mp4'
DEFAULT_VIDEO_2 = r'C:\Users\Nadhu\Downloads\stock-footage-uttarakhand-in-may-the-traffic-on-a-curvy-road-surrounded-by-mountains-under-blue-sky.mp4'
FOCAL_LENGTH  = None

# 1. Roadside ESP32 (Traffic Lights + Dual 1.3" OLEDs)
ESP32_IP   = "10.131.104.140"  # <-- Roadside ESP32 IP
ESP32_PORT = 4210

# 2. In-Car ESP32 Dev Module (Side 1 Car Alert: 0.96" OLED + Red LED + 1s Buzzer)
CAR_ESP32_IP   = "10.131.104.118"  # <-- කාරයේ ESP32 IP එක මෙතනට දාන්න
CAR_ESP32_PORT = 4210

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
#  COLOR HELPERS & OPTICAL FLOW
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

def dist_dot_char(d):
    if d is None: return '⚪'
    if d < 10:    return '🔴'
    if d < 25:    return '🟠'
    if d < 50:    return '🟡'
    return '🟢'

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
#  WARNING OVERLAY & SPEEDOMETER
# ═══════════════════════════════════════════════════════════

def draw_warning_overlay(frame, vehicles, base_safe, flash_state, flash_intensity, title="WARNING SYSTEM"):
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
    cv2.rectangle(frame, (x0,y0), (x0+pw,y0+ph), bcol, 3 if danger else 1)
    cv2.rectangle(frame, (x0,y0), (x0+pw,y0+22), (25,30,50), -1)
    cv2.putText(frame, f' {title}', (x0+4, y0+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (74,158,255), 1, cv2.LINE_AA)

    yt = y0 + 38
    if closest:
        d    = closest['distance']
        dcol = ((0,0,230) if d<10 else (0,115,250) if d<25
                else (0,220,230) if d<50 else (50,200,50))
        dlbl = ('DANGER' if d<10 else 'WARNING' if d<25
                else 'CAUTION' if d<50 else 'SAFE')
        cv2.circle(frame, (x0+16, yt-2), 7, dcol, -1)
        cv2.putText(frame, f'{dlbl}  Closest:{d:.1f}m ({closest["type"]})',
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
            cv2.putText(frame, f'  #{v["id"]} {v["type"]}  {int(v["speed"])}km/h  +{int(excess)} over',
                        (x0+4,yt+4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, scol, 1, cv2.LINE_AA)
            yt += 16

    if danger and flash_intensity > 0.02:
        border_overlay = frame.copy()
        cv2.rectangle(border_overlay, (0,0), (w-1,h-1), (0,0,200), 5)
        cv2.addWeighted(border_overlay, flash_intensity, frame, 1-flash_intensity, 0, frame)

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
        self.alert_logged=False
        self.is_towards=True

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
                self.is_towards = (d1 <= d0 + 0.5)
                sp_kh=min(abs(d1-d0)/dt*3.6,MAX_SPEED_KMPH)
                if not (self._raw_speeds and sp_kh>max(self._raw_speeds)*3 and len(self._raw_speeds)>=3):
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
#  SHARED STATE (DUAL SOURCE + DUAL ESP32 NODE)
# ═══════════════════════════════════════════════════════════

class SharedState:
    def __init__(self):
        self._lock=threading.Lock()
        self.frame1=None; self.frame2=None
        self.vehicles_1=[]; self.vehicles_2=[]
        self.confidence=0.40; self.friction=0.70
        self.radius=30.0; self.bank_angle=10.0
        self.base_safe=banked_speed(0.70,30.0,10.0)
        self.vehicle_limits={cls: vehicle_limit(cls, self.base_safe) for cls in CLASS_NAMES}
        self.overspeed=False; self.fps=0.0
        self.frame_count=0; self.total_frames=0
        self.max_speed_seen=0.0; self.max_speed_cls=''
        self.speed_alerts=0; self.running=True; self.paused=False

        self._src_lock=threading.Lock()
        self._new_source1=None; self._new_source2=None
        self.source_status1='idle'; self.source_status2='idle'

        self.esp32_connected=False; self.esp32_last_success=0.0
        self.car_esp32_connected=False; self.car_esp32_last_success=0.0
        self._alerts_log=deque(maxlen=30)

    def push(self, f1, f2, v1, v2, overspeed, fps, fc, mx_spd, mx_cls, alerts):
        with self._lock:
            self.frame1=f1.copy() if f1 is not None else None
            self.frame2=f2.copy() if f2 is not None else None
            self.vehicles_1=v1; self.vehicles_2=v2
            self.overspeed=overspeed
            self.fps=fps; self.frame_count=fc
            self.max_speed_seen=mx_spd; self.max_speed_cls=mx_cls
            self.speed_alerts=alerts

    def set_conf(self,v):
        with self._lock: self.confidence=v

    def set_curve(self,mu,r,b):
        with self._lock:
            self.friction=mu; self.radius=r; self.bank_angle=b
            self.base_safe=banked_speed(mu,r,b)
            self.vehicle_limits={cls: vehicle_limit(cls, self.base_safe) for cls in CLASS_NAMES}

    def request_source(self,source,side=1):
        with self._src_lock:
            if side==1: self._new_source1=source
            else: self._new_source2=source
        with self._lock:
            if side==1: self.source_status1='connecting'
            else: self.source_status2='connecting'

    def poll_source(self):
        with self._src_lock:
            s1, s2 = self._new_source1, self._new_source2
            self._new_source1, self._new_source2 = None, None
            return s1, s2

    def set_source_status(self,side,status):
        with self._lock:
            if side==1: self.source_status1=status
            else: self.source_status2=status

    def get_source_status(self):
        with self._lock: return self.source_status1, self.source_status2

    def set_esp32_status(self,connected):
        with self._lock:
            self.esp32_connected=connected
            if connected: self.esp32_last_success=time.time()

    def get_esp32_status(self):
        with self._lock: return self.esp32_connected, self.esp32_last_success

    def set_car_esp32_status(self,connected):
        with self._lock:
            self.car_esp32_connected=connected
            if connected: self.car_esp32_last_success=time.time()

    def get_car_esp32_status(self):
        with self._lock: return self.car_esp32_connected, self.car_esp32_last_success

    def push_alert(self,side,vid,cls_name,speed,limit):
        with self._lock:
            self._alerts_log.appendleft(
                (time.strftime('%H:%M:%S'), side, vid, cls_name, speed, limit))

    def get_alerts(self):
        with self._lock: return list(self._alerts_log)

    def get_frames(self):
        with self._lock:
            return (self.frame1.copy() if self.frame1 is not None else None,
                    self.frame2.copy() if self.frame2 is not None else None)

    def get_vehicles(self):
        with self._lock: return list(self.vehicles_1), list(self.vehicles_2)

    def get_curve_params(self):
        with self._lock:
            return (self.confidence,self.friction,self.radius,
                    self.bank_angle,self.base_safe)

    def get_vehicle_limits(self):
        with self._lock: return dict(self.vehicle_limits)

# ═══════════════════════════════════════════════════════════
#  VIDEO WORKER THREAD (DUAL STREAM + ROADSIDE & CAR UDP)
# ═══════════════════════════════════════════════════════════

def open_source(src, state, side):
    if src is None: return None, False, 0, 1/30.0
    try:
        cam_idx=int(src); is_cam=True
        cap=cv2.VideoCapture(cam_idx)
    except (ValueError,TypeError):
        is_cam=False; cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        state.set_source_status(side, 'failed')
        return None,False,0,1/30.0
    ok,_=cap.read()
    if not ok:
        cap.release()
        state.set_source_status(side, 'failed')
        return None,False,0,1/30.0
    if not is_cam: cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0
    vdt=1.0/max(cap.get(cv2.CAP_PROP_FPS) or 30.0,1)
    state.set_source_status(side, 'connected')
    return cap,is_cam,total,vdt

def process_side(model, frame, prev_gray, tracker, vdt, frame_cnt, focal, conf, base_safe, side_id, state, flash_state, flash_intensity):
    if frame is None: return None, None, [], False, 0.0, '', 0

    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = model(frame, conf=conf, iou=0.45, imgsz=640, device=device, verbose=False)[0]

    detections = []
    for b in res.boxes:
        cid = int(b.cls[0])
        cname = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else 'unknown'
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        detections.append(((x1, y1, x2, y2), cname))

    tracks = tracker.update(detections)
    overspeed = False; vehicles = []; alerts = 0
    max_spd = 0.0; max_cls = ''

    for trk in tracks:
        x1, y1, x2, y2 = trk.box; box_h = y2 - y1
        trk.push_distance(focal_distance(box_h, trk.cls_name, focal))
        trk.push_speed(vdt)

        if not getattr(trk, 'is_towards', True): continue

        if prev_gray is not None:
            if trk.flow_pts is None: trk.flow_pts = extract_feats(prev_gray, trk.box)
            mag = flow_mag(prev_gray, curr_gray, trk.flow_pts)
            trk.is_stationary = mag is not None and mag < STATIONARY_PX
            if frame_cnt % 15 == 0: trk.flow_pts = extract_feats(curr_gray, trk.box)
        if trk.is_stationary: trk.smooth_speed = max(trk.smooth_speed * 0.7, 0.0)

        veh_lim = vehicle_limit(trk.cls_name, base_safe)
        show_s = trk.frame_count >= MIN_TRACK_FRAMES

        if show_s and trk.smooth_speed > veh_lim:
            overspeed = True; alerts += 1
            if not trk.alert_logged:
                state.push_alert(side_id, trk.id, trk.cls_name, trk.smooth_speed, veh_lim)
                trk.alert_logged = True
        elif show_s:
            trk.alert_logged = False

        if trk.smooth_speed > max_spd: max_spd = trk.smooth_speed; max_cls = trk.cls_name

        color = CLASS_COLORS_BGR.get(trk.cls_name, (200, 200, 200))
        scol = spd_color_bgr(trk.smooth_speed, veh_lim)
        bcol = (0, 0, 255) if (show_s and trk.smooth_speed >= veh_lim) else color
        bthk = 3 if (show_s and trk.smooth_speed >= veh_lim) else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), bcol, bthk)

        dist_txt = f'{trk.display_dist:.1f}m' if trk.display_dist else '?m'
        lbl = f'S{side_id}-#{trk.id} {trk.cls_name} | {dist_txt}'
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, lbl, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        if show_s:
            stx = 'stationary' if trk.is_stationary and trk.smooth_speed < MIN_SPEED_KMPH else f'{int(trk.smooth_speed)} km/h'
            (sw, sh), _ = cv2.getTextSize(stx, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            cv2.rectangle(frame, (x1, y2), (x1 + sw + 6, y2 + sh + 10), (0, 0, 0), -1)
            cv2.putText(frame, stx, (x1 + 3, y2 + sh + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.58, scol, 2, cv2.LINE_AA)
            cv2.putText(frame, f'Lim:{veh_lim:.0f}', (x1 + 3, y2 + sh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 200, 0), 1, cv2.LINE_AA)
            draw_speedometer(frame, trk.smooth_speed, veh_lim, (x1 + x2) // 2, max(y1 - 48, 42), r=33)

        vehicles.append({
            'side': side_id, 'id': trk.id, 'type': trk.cls_name,
            'speed': trk.smooth_speed if show_s else None,
            'distance': trk.display_dist, 'limit': veh_lim,
            'brake_kn': trk.brake_force_n / 1000 if trk.brake_force_n > 0 else 0.0,
            'stationary': trk.is_stationary, 'show_speed': show_s,
        })

    draw_warning_overlay(frame, vehicles, base_safe, flash_state, flash_intensity, title=f"WARNING SYSTEM - SIDE {side_id}")
    return frame, curr_gray, vehicles, overspeed, max_spd, max_cls, alerts

def video_worker(state, model, init_src1, init_src2):
    tracker1, tracker2 = Tracker(), Tracker()
    prev_gray1, prev_gray2 = None, None
    fps_disp = 30.0; t_prev = time.time()
    frame_cnt = 0; flash_tick = 0; flash_state = False

    cap1, is_cam1, tot1, vdt1 = open_source(init_src1, state, 1)
    cap2, is_cam2, tot2, vdt2 = open_source(init_src2, state, 2)

    while state.running:
        req1, req2 = state.poll_source()
        if req1 is not None:
            if cap1: cap1.release()
            cap1, is_cam1, tot1, vdt1 = open_source(req1, state, 1)
            tracker1 = Tracker(); prev_gray1 = None
        if req2 is not None:
            if cap2: cap2.release()
            cap2, is_cam2, tot2, vdt2 = open_source(req2, state, 2)
            tracker2 = Tracker(); prev_gray2 = None

        if state.paused:
            time.sleep(0.05); continue

        ret1, frame1 = cap1.read() if cap1 else (False, None)
        ret2, frame2 = cap2.read() if cap2 else (False, None)

        # Loop videos if file ends
        if cap1 and not ret1 and not is_cam1:
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0); ret1, frame1 = cap1.read()
        if cap2 and not ret2 and not is_cam2:
            cap2.set(cv2.CAP_PROP_POS_FRAMES, 0); ret2, frame2 = cap2.read()

        if not ret1 and not ret2:
            time.sleep(0.01); continue

        frame_cnt += 1
        flash_tick = (flash_tick + 1) % 20
        if flash_tick == 0: flash_state = not flash_state
        flash_intensity = (math.sin(frame_cnt * 0.25) + 1.0) / 2.0 * 0.55 + 0.10

        conf, mu, radius, bank, base_safe = state.get_curve_params()

        W1 = frame1.shape[1] if frame1 is not None else 640
        W2 = frame2.shape[1] if frame2 is not None else 640
        focal1 = FOCAL_LENGTH or (W1 * 0.85); focal2 = FOCAL_LENGTH or (W2 * 0.85)

        # Process Side 1
        f1, pg1, v1, os1, ms1, mc1, a1 = process_side(model, frame1, prev_gray1, tracker1, vdt1, frame_cnt, focal1, conf, base_safe, 1, state, flash_state, flash_intensity)
        # Process Side 2
        f2, pg2, v2, os2, ms2, mc2, a2 = process_side(model, frame2, prev_gray2, tracker2, vdt2, frame_cnt, focal2, conf, base_safe, 2, state, flash_state, flash_intensity)

        prev_gray1 = pg1; prev_gray2 = pg2
        any_overspeed = os1 or os2
        total_alerts = a1 + a2
        max_speed_seen = max(ms1, ms2)
        max_cls_seen = mc1 if ms1 > ms2 else mc2

        t_now = time.time()
        fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(t_now - t_prev, 1e-4))
        t_prev = t_now

        if f1 is not None: cv2.putText(f1, f'FPS:{fps_disp:.1f}', (W1-110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
        if f2 is not None: cv2.putText(f2, f'FPS:{fps_disp:.1f}', (W2-110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)

        state.push(f1, f2, v1, v2, any_overspeed, fps_disp, frame_cnt, max_speed_seen, max_cls_seen, total_alerts)

        # ═══════════════════════════════════════════════════════════
        #  1. SEND UDP JSON DATA TO ROADSIDE ESP32 NODE (DUAL SIDE)
        # ═══════════════════════════════════════════════════════════
        def extract_worst(vehicles):
            valid_d = [v['distance'] for v in vehicles if v['distance'] is not None]
            c_dist = min(valid_d) if valid_d else 999.0
            ov_v = [v for v in vehicles if v['show_speed'] and v['speed'] and v['speed'] >= v['limit']]
            w_v = max(ov_v, key=lambda x: x['speed'] - x['limit']) if ov_v else None
            return c_dist, w_v

        cd1, wv1 = extract_worst(v1)
        cd2, wv2 = extract_worst(v2)

        payload = {
            "side1": {
                "dist": round(cd1, 1),
                "overspeed": bool(wv1),
                "limit": int(wv1['limit']) if wv1 else 0,
                "type": wv1['type'] if wv1 else ""
            },
            "side2": {
                "dist": round(cd2, 1),
                "overspeed": bool(wv2),
                "limit": int(wv2['limit']) if wv2 else 0,
                "type": wv2['type'] if wv2 else ""
            }
        }

        try:
            udp_socket.sendto(json.dumps(payload).encode('utf-8'), (ESP32_IP, ESP32_PORT))
            state.set_esp32_status(True)
        except Exception:
            state.set_esp32_status(False)

        # ═══════════════════════════════════════════════════════════
        #  2. SEND UDP JSON DATA TO IN-CAR ESP32 (SIDE 1 CAR ONLY)
        # ═══════════════════════════════════════════════════════════
        s1_cars = [v for v in v1 if v['type'] == 'car']
        s1_over_cars = [
            v for v in s1_cars
            if v['show_speed'] and v['speed'] is not None and v['speed'] >= v['limit']
        ]
        worst_car = max(s1_over_cars, key=lambda x: x['speed'] - x['limit']) if s1_over_cars else None

        car_payload = {
            "car_detected": bool(s1_cars),
            "overspeed": bool(worst_car),
            "limit": int(worst_car['limit']) if worst_car else 0
        }

        try:
            udp_socket.sendto(json.dumps(car_payload).encode('utf-8'), (CAR_ESP32_IP, CAR_ESP32_PORT))
            state.set_car_esp32_status(True)
        except Exception:
            state.set_car_esp32_status(False)

    if cap1: cap1.release()
    if cap2: cap2.release()

# ═══════════════════════════════════════════════════════════
#  TKINTER APPLICATION
# ═══════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self,state):
        super().__init__()
        self.state=state
        self.title('Vehicle Detection System  —  Dual View Curve Physics  v12 (V2I Car Alert)')
        self.configure(bg=BG)
        self.geometry('1500x860'); self.minsize(1300,740)
        self._last_vehicle_fp=None
        self._setup_styles()
        self._build_left_panel()
        self._build_notebook()
        self.protocol('WM_DELETE_WINDOW',self._on_close)
        self.bind('<space>',lambda e: self._toggle_pause())
        self._tick()

    def _setup_styles(self):
        s=ttk.Style(self); s.theme_use('clam')
        s.configure('.',background=BG,foreground=TEXT,font=('Segoe UI',10))
        s.configure('TNotebook',background=BG2,borderwidth=0)
        s.configure('TNotebook.Tab',background=BG3,foreground=TEXT2,padding=[16,7],font=('Segoe UI',10,'bold'))
        s.map('TNotebook.Tab',background=[('selected',BG)],foreground=[('selected',ACCENT)])
        s.configure('TFrame',background=BG)
        s.configure('TLabel',background=BG,foreground=TEXT)
        s.configure('TScale',background=BG2,troughcolor=BG3,sliderthickness=18)
        s.configure('Treeview',background=BG3,foreground=TEXT,fieldbackground=BG3,rowheight=28,font=('Segoe UI',9))
        s.configure('Treeview.Heading',background=BG2,foreground=ACCENT,font=('Segoe UI',9,'bold'),relief='flat')
        s.map('Treeview',background=[('selected',BG4)],foreground=[('selected',TEXT)])
        s.configure('TButton',background=BG3,foreground=TEXT,font=('Segoe UI',10),padding=[10,6],relief='flat')
        s.map('TButton',background=[('active',BG4)])
        s.configure('TCombobox',background=BG3,foreground=TEXT,fieldbackground=BG3,selectbackground=BG4,selectforeground=TEXT)
        s.configure('Accent.TButton',background=ACCENT,foreground='#000',font=('Segoe UI',10,'bold'),padding=[10,6])
        s.map('Accent.TButton',background=[('active','#3a8eef')])

    def _build_left_panel(self):
        lp=tk.Frame(self,bg=BG2,width=230)
        lp.pack(side='left',fill='y',padx=(8,4),pady=8)
        lp.pack_propagate(False)

        tk.Label(lp,text='🚦',font=('Segoe UI',28),bg=BG2).pack(pady=(14,2))
        tk.Label(lp,text='DUAL VEHICLE\nDETECTION',font=('Segoe UI',12,'bold'),fg=ACCENT,bg=BG2,justify='center').pack()
        self._hline(lp)

        tk.Label(lp,text='VIDEO SOURCE',font=('Segoe UI',9,'bold'),fg=TEXT2,bg=BG2).pack(pady=(4,4))

        ts_frame = tk.Frame(lp, bg=BG2)
        ts_frame.pack(fill='x', padx=10, pady=(0, 6))
        tk.Label(ts_frame, text='Target:', bg=BG2, fg=TEXT2, font=('Segoe UI', 9)).pack(side='left', padx=(0,5))
        self.target_side = tk.IntVar(value=1)
        tk.Radiobutton(ts_frame, text='Side 1', variable=self.target_side, value=1, bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2, activeforeground=ACCENT).pack(side='left', padx=2)
        tk.Radiobutton(ts_frame, text='Side 2', variable=self.target_side, value=2, bg=BG2, fg=TEXT, selectcolor=BG3, activebackground=BG2, activeforeground=ACCENT).pack(side='left', padx=2)

        self.src_type=tk.StringVar(value='file')
        rf=tk.Frame(lp,bg=BG2); rf.pack(fill='x',padx=10,pady=(0,6))
        for val,txt in [('camera','Camera'),('file','Video File')]:
            tk.Radiobutton(rf,text=txt,variable=self.src_type,value=val,bg=BG2,fg=TEXT,selectcolor=BG3,activebackground=BG2,activeforeground=ACCENT,command=self._on_src_type).pack(side='left',padx=4)

        self._src_container=tk.Frame(lp,bg=BG2,height=112)
        self._src_container.pack(fill='x',padx=10,pady=2)
        self._src_container.pack_propagate(False)
        self._src_container.grid_columnconfigure(0,weight=1)

        self.cam_frame=tk.Frame(self._src_container,bg=BG2)
        self.cam_frame.grid(row=0,column=0,sticky='nsew')
        cam_row=tk.Frame(self.cam_frame,bg=BG2); cam_row.pack(fill='x',pady=(4,2))
        tk.Label(cam_row,text='Cam index:',font=('Segoe UI',9),fg=TEXT2,bg=BG2).pack(side='left')
        self.cam_idx_var=tk.StringVar(value='0')
        self.cam_combo=ttk.Combobox(cam_row,textvariable=self.cam_idx_var,values=['0','1','2','3'],state='readonly',width=4)
        self.cam_combo.pack(side='left',padx=6)
        self.cam_detect_lbl=tk.Label(self.cam_frame,text='',font=('Segoe UI',8),fg=TEXT2,bg=BG2,anchor='w')
        self.cam_detect_lbl.pack(fill='x')
        ttk.Button(self.cam_frame,text='🔍 Detect Cameras',command=self._detect_cameras).pack(fill='x',pady=(2,0))

        self.file_frame=tk.Frame(self._src_container,bg=BG2)
        self.file_frame.grid(row=0,column=0,sticky='nsew')

        self._full_file_path = DEFAULT_VIDEO_1
        self.file_path_var = tk.StringVar(value=os.path.basename(self._full_file_path))
        tk.Label(self.file_frame,textvariable=self.file_path_var,font=('Segoe UI',8),fg=TEXT2,bg=BG2,wraplength=190,justify='left',anchor='w').pack(fill='x')
        ttk.Button(self.file_frame,text='📂 Browse…',command=self._browse_file).pack(fill='x',pady=(4,0))
        self._on_src_type()

        ttk.Button(lp,text='▶ Connect to Side',style='Accent.TButton',command=self._connect_source).pack(padx=14,pady=(8,2),fill='x')
        self.source_status_lbl=tk.Label(lp,text='S1: connected | S2: connected',font=('Segoe UI',8,'bold'),fg=TEXT2,bg=BG2)
        self.source_status_lbl.pack(padx=14,pady=(0,6),anchor='w')
        self._hline(lp)

        tk.Label(lp,text='LIVE STATS',font=('Segoe UI',9,'bold'),fg=TEXT2,bg=BG2).pack(pady=(4,6))
        self.stats={}
        for key,lbl,default,col in [
            ('vehicles','Total Veh','0',TEXT),
            ('fps','FPS','0.0',ACCENT),
            ('max_speed','Max Speed','0 km/h',TEXT),
            ('alerts','Alerts','0',TEXT),
        ]:
            row=tk.Frame(lp,bg=BG2); row.pack(fill='x',padx=10,pady=2)
            tk.Label(row,text=lbl,font=('Segoe UI',9),fg=TEXT2,bg=BG2,anchor='w',width=10).pack(side='left')
            w=tk.Label(row,text=default,font=('Segoe UI',9,'bold'),fg=col,bg=BG2,anchor='e')
            w.pack(side='right'); self.stats[key]=w
        self._hline(lp)

        # Roadside + In-Car ESP32 Link Status
        esp_row=tk.Frame(lp,bg=BG2); esp_row.pack(fill='x',padx=14,pady=(0,2))
        self.esp32_dot=tk.Label(esp_row,text='●',font=('Segoe UI',11),fg=TEXT2,bg=BG2)
        self.esp32_dot.pack(side='left')
        tk.Label(esp_row,text=' Roadside ESP32',font=('Segoe UI',9,'bold'),fg=TEXT2,bg=BG2).pack(side='left')
        self.esp32_status_lbl=tk.Label(lp,text='no data sent yet',font=('Segoe UI',8),fg=TEXT2,bg=BG2,wraplength=200,justify='left',anchor='w')
        self.esp32_status_lbl.pack(fill='x',padx=14,pady=(0,4))

        car_row=tk.Frame(lp,bg=BG2); car_row.pack(fill='x',padx=14,pady=(0,2))
        self.car_esp32_dot=tk.Label(car_row,text='●',font=('Segoe UI',11),fg=TEXT2,bg=BG2)
        self.car_esp32_dot.pack(side='left')
        tk.Label(car_row,text=' In-Car ESP32 (S1)',font=('Segoe UI',9,'bold'),fg=TEXT2,bg=BG2).pack(side='left')
        self.car_esp32_status_lbl=tk.Label(lp,text='no data sent yet',font=('Segoe UI',8),fg=TEXT2,bg=BG2,wraplength=200,justify='left',anchor='w')
        self.car_esp32_status_lbl.pack(fill='x',padx=14,pady=(0,4))
        self._hline(lp)

        self.pause_btn=ttk.Button(lp,text='⏸ Pause (Space)',command=self._toggle_pause)
        self.pause_btn.pack(padx=14,pady=(4,10),fill='x')
        quit_frame=tk.Frame(lp,bg=BG2); quit_frame.pack(padx=14,pady=(0,4),fill='x')
        tk.Button(quit_frame,text='✕ Quit',command=self._on_close,bg='#3a1414',fg=C_RED,activebackground='#4a1a1a',activeforeground=C_RED,relief='flat',font=('Segoe UI',10,'bold'),padx=10,pady=6).pack(fill='x')

    def _on_src_type(self):
        if self.src_type.get()=='camera': self.cam_frame.tkraise()
        else: self.file_frame.tkraise()

    def _browse_file(self):
        path=filedialog.askopenfilename(title='Select Video File',filetypes=[('Video files','*.mp4 *.avi *.mkv *.mov *.m4v *.wmv *.webm'),('All files','*.*')])
        if path:
            self._full_file_path=path
            self.file_path_var.set(os.path.basename(path))

    def _detect_cameras(self):
        self.cam_detect_lbl.config(text='Scanning indices 0–3…',fg=C_YELLOW); self.update_idletasks()
        found=[]
        for idx in range(4):
            cap=cv2.VideoCapture(idx)
            if cap.isOpened():
                ok,_=cap.read()
                if ok: found.append(str(idx))
            cap.release()
        if found:
            self.cam_combo.config(values=found); self.cam_idx_var.set(found[0])
            self.cam_detect_lbl.config(text=f'Found: {", ".join(found)}',fg=C_GREEN)
        else: self.cam_detect_lbl.config(text='No cameras found',fg=C_RED)

    def _connect_source(self):
        src = self.cam_idx_var.get() if self.src_type.get()=='camera' else self._full_file_path
        if not src:
            messagebox.showwarning('No file','Please select a source.')
            return
        side = self.target_side.get()
        self._last_vehicle_fp = None
        self.state.request_source(src, side)

    def _build_notebook(self):
        self.nb=ttk.Notebook(self)
        self.nb.pack(side='left',fill='both',expand=True,padx=(0,8),pady=8)
        self.tab1=ttk.Frame(self.nb)
        self.nb.add(self.tab1,text='  📊 Dual Camera Feed  ')
        self._build_tab1()
        self.tab2=ttk.Frame(self.nb)
        self.nb.add(self.tab2,text='  ⚙ Curve Settings  ')
        self._build_tab2()

    def _build_tab1(self):
        main = tk.Frame(self.tab1, bg=BG)
        main.pack(fill='both', expand=True)
        main.columnconfigure(0, weight=1, uniform='cols')
        main.columnconfigure(1, weight=1, uniform='cols')
        main.rowconfigure(0, weight=1)

        # ---------------------------------------------------------
        # COLUMN 1 (SIDE 1)
        # ---------------------------------------------------------
        col1 = tk.Frame(main, bg=BG2)
        col1.grid(row=0, column=0, sticky='nsew', padx=(0, 3))

        hdr1 = tk.Frame(col1, bg=BG2); hdr1.pack(fill='x', padx=6, pady=(8, 2))
        tk.Label(hdr1, text=' SIDE 1 FEED', font=('Segoe UI', 11, 'bold'), fg=ACCENT, bg=BG2, anchor='w').pack(side='left')

        self.video_canvas1 = tk.Canvas(col1, bg='#000', highlightthickness=1, highlightbackground=BG3)
        self.video_canvas1.pack(side='top', fill='both', expand=True, padx=6, pady=2)

        data1 = tk.Frame(col1, bg=BG2, height=180)
        data1.pack(side='bottom', fill='x', padx=6, pady=6)
        data1.pack_propagate(False)

        cols = ('ID', 'Type', 'Speed', 'Dist', 'Limit', 'Brake')
        widths = {'ID': 40, 'Type': 70, 'Speed': 65, 'Dist': 60, 'Limit': 50, 'Brake': 55}

        self.tree1 = ttk.Treeview(data1, columns=cols, show='headings', height=5)
        for c in cols:
            self.tree1.heading(c, text=c)
            self.tree1.column(c, width=widths[c], anchor='center', minwidth=widths[c])
        vsb1 = ttk.Scrollbar(data1, orient='vertical', command=self.tree1.yview)
        self.tree1.configure(yscrollcommand=vsb1.set)
        self.tree1.pack(side='left', fill='both', expand=True)
        vsb1.pack(side='left', fill='y')

        self.alerts_list1 = tk.Listbox(data1, bg=BG3, fg=TEXT2, font=('Consolas', 8), width=32, borderwidth=0, highlightthickness=0, selectbackground=BG4)
        self.alerts_list1.pack(side='right', fill='y', padx=(4, 0))

        # ---------------------------------------------------------
        # COLUMN 2 (SIDE 2)
        # ---------------------------------------------------------
        col2 = tk.Frame(main, bg=BG2)
        col2.grid(row=0, column=1, sticky='nsew', padx=(3, 0))

        hdr2 = tk.Frame(col2, bg=BG2); hdr2.pack(fill='x', padx=6, pady=(8, 2))
        tk.Label(hdr2, text=' SIDE 2 FEED', font=('Segoe UI', 11, 'bold'), fg=ACCENT, bg=BG2, anchor='w').pack(side='left')

        self.video_canvas2 = tk.Canvas(col2, bg='#000', highlightthickness=1, highlightbackground=BG3)
        self.video_canvas2.pack(side='top', fill='both', expand=True, padx=6, pady=2)

        data2 = tk.Frame(col2, bg=BG2, height=180)
        data2.pack(side='bottom', fill='x', padx=6, pady=6)
        data2.pack_propagate(False)

        self.tree2 = ttk.Treeview(data2, columns=cols, show='headings', height=5)
        for c in cols:
            self.tree2.heading(c, text=c)
            self.tree2.column(c, width=widths[c], anchor='center', minwidth=widths[c])
        vsb2 = ttk.Scrollbar(data2, orient='vertical', command=self.tree2.yview)
        self.tree2.configure(yscrollcommand=vsb2.set)
        self.tree2.pack(side='left', fill='both', expand=True)
        vsb2.pack(side='left', fill='y')

        self.alerts_list2 = tk.Listbox(data2, bg=BG3, fg=TEXT2, font=('Consolas', 8), width=32, borderwidth=0, highlightthickness=0, selectbackground=BG4)
        self.alerts_list2.pack(side='right', fill='y', padx=(4, 0))

        self._vimg1 = None; self._vimg2 = None
        self._alerts_seen = 0

    def _build_tab2(self):
        outer=tk.Frame(self.tab2,bg=BG); outer.pack(fill='both',expand=True,padx=24,pady=16)
        tk.Label(outer,text='Mountain Curve Physics Settings',font=('Segoe UI',15,'bold'),fg=ACCENT,bg=BG).pack(pady=(0,18))
        row=tk.Frame(outer,bg=BG); row.pack(fill='x',pady=6)
        tk.Label(row,text='Road Surface:',font=('Segoe UI',10),fg=TEXT2,bg=BG,width=26,anchor='w').pack(side='left')
        self.surface_var=tk.StringVar(value='Dry Asphalt')
        cb=ttk.Combobox(row,textvariable=self.surface_var,values=list(ROAD_CONDITIONS.keys()),state='readonly',width=18)
        cb.pack(side='left',padx=8)
        cb.bind('<<ComboboxSelected>>',self._on_surface)

        self._add_slider(outer,'Friction Coefficient (μ)','mu_var','mu_lbl',0.05,0.90,0.70,2)
        self._add_slider(outer,'Curve Radius (m)','rad_var','rad_lbl',10.0,200.0,30.0,1)
        self._add_slider(outer,'Bank Angle (°)','bank_var','bank_lbl',0.0,30.0,10.0,1)
        self._add_slider(outer,'Detection Confidence','conf_var','conf_lbl',0.05,0.95,self.state.confidence,2,self._on_conf)
        self._hline(outer,bg=BG)

        tk.Label(outer,text='MAX SAFE SPEED PER VEHICLE TYPE',font=('Segoe UI',10,'bold'),fg=TEXT2,bg=BG).pack(pady=(0,2))

        grid_frame=tk.Frame(outer,bg=BG); grid_frame.pack(fill='x',pady=(0,4))
        COLS=3
        for i in range(COLS): grid_frame.columnconfigure(i,weight=1)
        self._mss_labels={}
        for idx,cls in enumerate(CLASS_NAMES):
            col=idx % COLS; row_i=idx // COLS
            card=tk.Frame(grid_frame,bg=BG3,relief='flat',bd=0)
            card.grid(row=row_i,column=col,padx=4,pady=4,sticky='ew')
            tk.Label(card,text=cls,font=('Segoe UI',8,'bold'),fg=TEXT2,bg=BG3,anchor='w').pack(fill='x',padx=8,pady=(6,0))
            spd_lbl=tk.Label(card,text='— km/h',font=('Segoe UI',14,'bold'),fg=C_GREEN,bg=BG3,anchor='w')
            spd_lbl.pack(fill='x',padx=8,pady=(2,4))
            bar=tk.Canvas(card,height=4,bg=BG4,highlightthickness=0,bd=0)
            bar.pack(fill='x',padx=8,pady=(0,6))
            self._mss_labels[cls]=(spd_lbl,bar)
        self._refresh_mss()

    def _add_slider(self,parent,label,var_name,lbl_name,lo,hi,init,dec,callback=None):
        row=tk.Frame(parent,bg=BG); row.pack(fill='x',pady=8)
        tk.Label(row,text=label,font=('Segoe UI',10),fg=TEXT2,bg=BG,width=28,anchor='w').pack(side='left')
        var=tk.DoubleVar(value=init); setattr(self,var_name,var)
        val_lbl=tk.Label(row,text=f'{init:.{dec}f}',font=('Segoe UI',12,'bold'),fg=ACCENT,bg=BG,width=7,anchor='e')
        val_lbl.pack(side='right',padx=8); setattr(self,lbl_name,val_lbl)
        fmt=f'{{:.{dec}f}}'
        cb=callback or self._on_curve
        def on_slide(v,L=val_lbl,f=fmt,fn=cb): L.config(text=f.format(float(v))); fn()
        ttk.Scale(row,from_=lo,to=hi,variable=var,orient='horizontal',length=280,command=on_slide).pack(side='left',padx=8)

    def _on_surface(self,_=None):
        mu=ROAD_CONDITIONS.get(self.surface_var.get(),0.70)
        self.mu_var.set(mu); self.mu_lbl.config(text=f'{mu:.2f}')
        self._on_curve()

    def _on_curve(self):
        self.state.set_curve(self.mu_var.get(),self.rad_var.get(),self.bank_var.get())
        self._refresh_mss()

    def _on_conf(self):
        self.state.set_conf(self.conf_var.get())

    def _toggle_pause(self):
        self.state.paused=not self.state.paused
        self.pause_btn.config(text='▶ Resume' if self.state.paused else '⏸ Pause')

    def _on_close(self):
        if messagebox.askyesno('Quit','Stop detection and close the application?'):
            self.state.running=False; self.destroy()

    def _refresh_mss(self):
        if not hasattr(self,'mu_var'): return
        base=banked_speed(self.mu_var.get(),self.rad_var.get(),self.bank_var.get())
        for cls,(spd_lbl,bar) in self._mss_labels.items():
            limit=vehicle_limit(cls,base)
            if limit==float('inf'): txt='∞ km/h'; fg=C_GREEN
            else:
                txt=f'{limit:.1f} km/h'
                fg=(C_GREEN if limit>50 else C_YELLOW if limit>30 else C_RED)
            spd_lbl.config(text=txt,fg=fg)
            self.after(10, lambda b=bar, lim=limit, c=fg: self._draw_bar(b,lim,c))

    def _draw_bar(self,bar,limit,color):
        bar.update_idletasks()
        w = bar.winfo_width()
        if w < 4: return
        bar.delete('all')
        bar.create_rectangle(0,0,w,4,fill=BG4,outline='')
        frac=min(limit/MAX_SPEED_KMPH,1.0) if limit!=float('inf') else 1.0
        filled=max(4,int(w*frac))
        bar.create_rectangle(0,0,filled,4,fill=color,outline='')

    def _tick(self):
        self._refresh_video()
        self._refresh_table()
        self._refresh_stats()
        self._refresh_link_status()
        self._refresh_alerts_log()
        self.after(40,self._tick)

    def _refresh_link_status(self):
        s1, s2 = self.state.get_source_status()
        t1 = "●" if s1=='connected' else "○"
        t2 = "●" if s2=='connected' else "○"
        self.source_status_lbl.config(text=f'S1: {t1} {s1}  |  S2: {t2} {s2}', fg=C_GREEN if (s1=='connected' or s2=='connected') else TEXT2)

        # Roadside ESP32 status
        connected, last_ok = self.state.get_esp32_status()
        age = time.time() - last_ok if last_ok else None
        if last_ok == 0.0:
            self.esp32_dot.config(fg=TEXT2); self.esp32_status_lbl.config(text='no data sent yet')
        elif connected and age is not None and age < 3.0:
            self.esp32_dot.config(fg=C_GREEN)
            self.esp32_status_lbl.config(text=f'{ESP32_IP}:{ESP32_PORT} (ok {age:.1f}s)')
        else:
            self.esp32_dot.config(fg=C_RED)
            self.esp32_status_lbl.config(text=f'failing — {ESP32_IP}:{ESP32_PORT}')

        # In-Car ESP32 status
        car_conn, car_ok = self.state.get_car_esp32_status()
        car_age = time.time() - car_ok if car_ok else None
        if car_ok == 0.0:
            self.car_esp32_dot.config(fg=TEXT2); self.car_esp32_status_lbl.config(text='no data sent yet')
        elif car_conn and car_age is not None and car_age < 3.0:
            self.car_esp32_dot.config(fg=C_GREEN)
            self.car_esp32_status_lbl.config(text=f'{CAR_ESP32_IP}:{CAR_ESP32_PORT} (ok {car_age:.1f}s)')
        else:
            self.car_esp32_dot.config(fg=C_RED)
            self.car_esp32_status_lbl.config(text=f'failing — {CAR_ESP32_IP}:{CAR_ESP32_PORT}')

    def _update_canvas(self, canvas, frame, current_img):
        if frame is None: return current_img
        cw, ch = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        h, w = frame.shape[:2]
        if w<=0 or h<=0: return current_img
        scale = min(cw/w, ch/h)
        nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
        resized = cv2.resize(frame, (nw, nh))
        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        canvas.delete('all')
        canvas.create_image(cw//2, ch//2, anchor='center', image=img)
        return img

    def _refresh_video(self):
        f1, f2 = self.state.get_frames()
        self._vimg1 = self._update_canvas(self.video_canvas1, f1, self._vimg1)
        self._vimg2 = self._update_canvas(self.video_canvas2, f2, self._vimg2)

    def _refresh_table(self):
        v1, v2 = self.state.get_vehicles()

        def fp(v):
            spd = int(v['speed']) if v['speed'] is not None and v['show_speed'] else -1
            return (v['id'], spd, round(v['distance'] or 999, 0))

        cur_fp = frozenset((1, *fp(v)) for v in v1) | frozenset((2, *fp(v)) for v in v2)
        if cur_fp == self._last_vehicle_fp: return
        self._last_vehicle_fp = cur_fp

        def update_tree(tree, vehicles):
            for r in tree.get_children(): tree.delete(r)
            ordered = sorted(vehicles, key=lambda x: x['distance'] or 999)

            for v in ordered:
                spd = v['speed']; dist = v['distance']; lim = v['limit']; brk = v['brake_kn']; show = v['show_speed']
                spd_txt = f"{int(spd)} km/h" if spd is not None and show else 'tracking…'
                dist_txt = f"{dist:.1f} m" if dist else '?'
                brk_txt = f"{brk:.1f} kN" if brk > 0 else '—'
                tag = ('over' if spd and show and spd >= lim else 'caution' if spd and show and spd >= lim*.9 else 'ok')

                tree.insert('', 'end', values=(f'#{v["id"]}', v['type'], spd_txt, dist_txt, f'{lim:.0f}', brk_txt), tags=(tag,))

            tree.tag_configure('over', foreground=C_RED)
            tree.tag_configure('caution', foreground=C_YELLOW)
            tree.tag_configure('ok', foreground=TEXT)

        update_tree(self.tree1, v1)
        update_tree(self.tree2, v2)

    def _refresh_stats(self):
        v1, v2 = self.state.get_vehicles()
        self.stats['vehicles'].config(text=str(len(v1) + len(v2)))
        self.stats['fps'].config(text=f'{self.state.fps:.1f}')
        self.stats['max_speed'].config(
            text=f'{int(self.state.max_speed_seen)} km/h',
            fg=spd_color_tk(self.state.max_speed_seen, vehicle_limit(self.state.max_speed_cls, self.state.base_safe) if self.state.max_speed_cls else 99))
        self.stats['alerts'].config(text=str(self.state.speed_alerts), fg=C_RED if self.state.speed_alerts>0 else C_GREEN)

    def _refresh_alerts_log(self):
        entries = self.state.get_alerts()
        if len(entries) == self._alerts_seen: return
        self._alerts_seen = len(entries)

        self.alerts_list1.delete(0, 'end')
        self.alerts_list2.delete(0, 'end')

        has_alerts1 = False
        has_alerts2 = False

        for ts, side, vid, cls_name, speed, limit in entries:
            txt = f'{ts} | #{vid} {cls_name[:6]} | {speed:.0f}>{limit:.0f}'
            if side == 1:
                self.alerts_list1.insert('end', txt)
                has_alerts1 = True
            else:
                self.alerts_list2.insert('end', txt)
                has_alerts2 = True

        if not has_alerts1: self.alerts_list1.insert('end', ' No alerts yet')
        if not has_alerts2: self.alerts_list2.insert('end', ' No alerts yet')

    def _hline(self, parent, bg=BG2):
        tk.Frame(parent, bg=BG3, height=1).pack(fill='x', pady=8, padx=8)

# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════

def main():
    print('\n'+'='*70)
    print('  Vehicle Detection System  —  Dual Camera + In-Car V2I  v12')
    print('='*70)
    model=YOLO(DEFAULT_MODEL)
    print('✅  Model loaded')
    state=SharedState()
    t=threading.Thread(target=video_worker, args=(state,model,DEFAULT_VIDEO_1,DEFAULT_VIDEO_2),daemon=True)
    t.start()
    app=App(state)
    app.mainloop()
    state.running=False
    udp_socket.close()
    print('Done.')

if __name__=='__main__':
    main()
