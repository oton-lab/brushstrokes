import cv2, csv
import numpy as np
import math

# ================== Parameters ==================
S_THRESH_L       = 0.7            
ON_HOLD = 2            # requires 2 consecutive frames with contact
OFF_HOLD = 2

def pca_orientation_deg_xy(points_xy):
    """
    Calculate the orientation (in degrees) of the major axis via PCA.
    points_xy: Nx2 with columns [x, y] (coordinates in pixels).
    Return degrees in [-90, 90] relative to the X-axis, CCW.
    """
    if points_xy.shape[0] < 2:
        return None

    # Center
    mean = points_xy.mean(axis=0)
    pts = points_xy - mean

    # Covariance and eigenvectors
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)       
    v_max = eigvecs[:, np.argmax(eigvals)]       

    # Angle respect X (atan2(y, x))
    angle_rad = math.atan2(v_max[1], v_max[0])
    angle_deg = math.degrees(angle_rad)

    # Normalize
    while angle_deg > 90:
        angle_deg -= 180
    while angle_deg <= -90:
        angle_deg += 180
    return angle_deg

# ================== Main program ==================
# --- open video ---
cap = cv2.VideoCapture('brushstrokes.mp4')
# Reduce latency in some files
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
# --- open CSV ---
csv_path = 'strokes.csv'
f = open(csv_path, 'w', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(['id', 'frames', 'time', 'alpha','beta','x','y','z'])  # header

# Counter status
contact = False
contact_count = 0
on_streak = 0
off_streak = 0
frames_count = 0
angle = 0
angleR = 0
xx = 0
yy = 0
zz = 0

while (cap.isOpened()):
  ret, frame = cap.read()
  msec = cap.get(cv2.CAP_PROP_POS_MSEC)   # ms
  secs = msec / 1000.0
  mm, ss = divmod(secs, 60)
  txt_mmss = f"t = {int(mm):02d}:{ss:04.1f}"  # mm:ss.s 
  cv2.putText(frame, txt_mmss,  (12, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,255), 1, cv2.LINE_AA)
  
  if ret == True:
    # --- Channel S (HSV) and binarize the brush hair -----------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    S = hsv[:,:,1]  # 0..255
    S1 = S[:,:479]  # left side
    _, bw = cv2.threshold(S1, int(S_THRESH_L*255), 255, cv2.THRESH_BINARY)
    V = hsv[:,:,2]  # 0..255

    # ---------- Binarize the black ferrule, LEFT : V < 40 ----------
    V1 = V[:,:479]
    V2 = V[:,480:]
    bw2 = (V1 < 40).astype(np.uint8)*255  # 0/1 
    # ---------- Binarize the black ferrule, RIGHT : V < 80 ----------
    bw_R = (V2 < 80).astype(np.uint8)*255  # 0/1
    bw_R[:, :40] = 0 # region of interest
    bw_R[:, 400:] = 0

    # x
    x = None
    angle_deg_R = 0
    yR, xR = np.where(bw_R == 255)

    if xR.size > 0:
      pts_xy_R = np.column_stack([xR, yR]).astype(np.float64)
      angle_deg_R = pca_orientation_deg_xy(pts_xy_R)
      yr = int(yR.min())
      x_candidates = xR[yR == yr]
      if x_candidates.size > 0:
        x = int(np.round(x_candidates.mean()))  # centroid

    # ---------- Remove small objects (area < 200) --------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw2, connectivity=8)
    # stats: [label, x, y, w, h, area] (area in stats[:, cv2.CC_STAT_AREA])
    min_area = 200
    bw3 = np.zeros_like(bw)
    for lab in range(1, num_labels):  
        if stats[lab, cv2.CC_STAT_AREA] >= min_area:
            bw3[labels == lab] = 1
    bw3[214:,:] = 0

    # ---------- "regionprops(...,'Orientation')" ----------
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(bw3, connectivity=8)
    angle_deg = 0
    y = None
    z = None
    ys, xs = np.where(labels2 == 1)
    if xs.size >= 3:
        pts_xy = np.column_stack([xs, ys]).astype(np.float64)
        angle_deg = pca_orientation_deg_xy(pts_xy)
        z = int(ys.max())
        x_candidates = xs[ys == z]
        if x_candidates.size > 0:
            y = int(np.round(x_candidates.mean()))  # centroid
    

    bw[:200, :] = 0 # region of interest
    bw[214:, :] = 0 
    
    obj = cv2.countNonZero(bw)
    
    is_contact = False
    
    if obj > 2:
      is_contact = True
      frames_count += 1
      angle += float(angle_deg)
      if angle_deg_R is not None:
        angleR += float(angle_deg_R)
      if x is not None:
        xx += int(x) 
      if y is not None:
        yy += int(y)     
      if z is not None:
        zz += int(z)     

    # --- Anti-bounce and counter increment on rising edge ---
    if is_contact:
        on_streak  += 1
        off_streak  = 0
    else:
        off_streak += 1
        on_streak   = 0

    if (not contact) and (on_streak >= 1):
        contact = True
        contact_count += 1          # <-- ¡new contact detected!
        print(f'Brushstroke: {contact_count}')
        
    elif contact and (off_streak >= 1):
        contact = False
        angle = -1*angle / frames_count
        angleR = angleR / frames_count
        xx = int(xx / frames_count)
        yy = int(yy / frames_count)
        zz = int(zz / frames_count)
        cv2.putText(frame, f'frames: {frames_count}', (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'angle: {angle:.2f}', (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'angleR: {angleR:.2f}', (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'x: {xx}', (12, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'y: {yy}', (12, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'z: {zz}', (12, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
        writer.writerow([contact_count, frames_count, f'{frames_count*33.33:.2f}',f'{angle:.2f}',f'{angleR:.2f}',xx,yy,zz])
        f.flush()  # save immediately
        frames_count = 0
        angle = 0
        angleR = 0
        xx = 0
        yy = 0
        zz = 0

    cv2.putText(frame, f'stroke id: {contact_count}', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow('video', frame)
    cv2.imshow('bw_left = side view', bw3*255)
    cv2.imshow('bw_right = top view', bw_R)

    if cv2.waitKey(30) == ord('q'):
      break

  else: break
# --- end while ---

cap.release()
cv2.destroyAllWindows()
f.close()
print(f'CSV saved in: {csv_path}')

