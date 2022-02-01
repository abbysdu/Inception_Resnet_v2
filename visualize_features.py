import cv2
import numpy as np
def visualize_features(feats,bw = None):
    if len(feats.shape) == 4:
        feats = feats[0]
    C,H,W = feats.shape
    nc = int(np.sqrt(C))
    nr = C//nc 
    if nr*nc<C:nr+=1
    margin = 2
    out = np.zeros((nr*H+(nr-1)*margin,nc*W+(nc-1)*margin,3))
    for c in range(C):
        h = feats[c]
        a = h.min()
        b = h.max()
        if a<b:
            h = (h-a)/(b-a)*255
            h = np.array(h,np.uint8)
            h = cv2.applyColorMap(h,cv2.COLORMAP_VIRIDIS)
        else:
            hh = np.zeros((h.shape[0],h.shape[1],3))
            hh[:,:,0] = h
            hh[:,:,1] = h
            hh[:,:,2] = h
            h = hh
        cr = c//nc
        cc = c - cr*nc
        sr = cr*(H+margin)
        sc = cc*(W+margin)
        if bw is not None:
            h[bw==0]=0
        out[sr:sr+H,sc:sc+W] = h
    return out