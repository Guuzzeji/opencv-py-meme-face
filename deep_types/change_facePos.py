def change(img2_head_mask, img, cv2, convexhull, result):
    
    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    return seamlessclone