import cv2
import numpy as np
import dlib


def scale(x, feature_range=(-1, 1)):
    min, max = feature_range
    x = x * (max - min) + min
    return x


def scale_back(x):
    min, max = feature_range
    x = (x + 1) / 2 * 255
    return x


def swap_faces(images_Y, images_X, batch_size):
    imgs_Y = images_Y
    idxs = torch.arange(imgs_Y.size(0))
    results = torch.tensor([])
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    imgs1 = images_Y.permute(0, 2, 3, 1)
    imgs2 = images_X.permute(0, 2, 3, 1)
    for i in range(batch_size):
        print(idxs)
        img1 = np.array(imgs1[i])
        img2 = np.array(imgs2[i])

        img1 = scale_back(img1, feature_range=(0, 255)).astype('uint8')
        img2 = scale_back(img2, feature_range=(0, 255)).astype('uint8')
        
#         img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#         img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        img2_new_face = np.zeros_like(img2, np.uint8)

        face_detector = dlib.get_frontal_face_detector()
        face_landmarks_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        faces = face_detector(img1_gray)
        faces2 = face_detector(img2_gray)

        if len(faces) == 0 or len(faces2) == 0:
            print('No faces detected')
            idxs = idxs[idxs!=i]
            continue
#             raise Exception('No faces detected')

        # Get face landmarks of first image
        for face in faces:
            landmarks = face_landmarks_predictor(img1_gray, face)
            landmarks_points = []
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))

            # Build convexhull given the face landmarks
            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)

            # Delaunay triangulation
            rect = cv2.boundingRect(convexhull)
            subdiv = cv2.Subdiv2D(rect)
            subdiv.insert(landmarks_points)
            triangles = subdiv.getTriangleList()
            triangles = np.array(triangles, dtype=np.int32)

            triangles_indexes = []
            for t in triangles:
                pt1, pt2, pt3 = (t[0], t[1]), (t[2], t[3]), (t[4], t[5])
                index_pt1 = np.where((points == pt1).all(axis=1))[0][0]
                index_pt2 = np.where((points == pt2).all(axis=1))[0][0]
                index_pt3 = np.where((points == pt3).all(axis=1))[0][0]
                if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                    triangle = [index_pt1, index_pt2, index_pt3]
                    triangles_indexes.append(triangle)

        # Get face landmarks of second image
        for face in faces2:
            landmarks2 = face_landmarks_predictor(img2_gray, face)
            landmarks_points2 = []
            for n in range(0, 68):
                x = landmarks2.part(n).x
                y = landmarks2.part(n).y
                landmarks_points2.append((x, y))

            points2 = np.array(landmarks_points2, np.int32)
            convexhull2 = cv2.convexHull(points2)
            
        try:
            for triangle_index in triangles_indexes:
                # Triangulation of the first face
                tr1_pt1 = landmarks_points[triangle_index[0]]
                tr1_pt2 = landmarks_points[triangle_index[1]]
                tr1_pt3 = landmarks_points[triangle_index[2]]
                triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
                (x, y, w, h) = cv2.boundingRect(triangle1)
                cropped_triangle = img1[y: y + h, x: x + w]
                cropped_tr1_mask = np.zeros((h, w), np.uint8)
                points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                                   [tr1_pt2[0] - x, tr1_pt2[1] - y],
                                   [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
                cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
                cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle,
                                                   mask=cropped_tr1_mask)

                # Triangulation of second face
                tr2_pt1 = landmarks_points2[triangle_index[0]]
                tr2_pt2 = landmarks_points2[triangle_index[1]]
                tr2_pt3 = landmarks_points2[triangle_index[2]]
                triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
                (x, y, w, h) = cv2.boundingRect(triangle2)
                cropped_triangle2 = img2[y: y + h, x: x + w]
                cropped_tr2_mask = np.zeros((h, w), np.uint8)
                points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                                    [tr2_pt2[0] - x, tr2_pt2[1] - y],
                                    [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
                cropped_triangle2 = cv2.bitwise_and(cropped_triangle2, cropped_triangle2,
                                                   mask=cropped_tr2_mask)

                # Warp triangles
                points = np.float32(points)
                points2 = np.float32(points2)
                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                # Reconstructing destination face
                img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
                img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 0, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
                img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
                img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area

            # Face swapped (putting 1st face into 2nd face)
            img2_face_mask = np.zeros_like(img2_gray)
            img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
            img2_face_mask = cv2.bitwise_not(img2_head_mask)

            img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
            result = cv2.add(img2_head_noface, img2_new_face)
            result = torch.from_numpy(result).float()
            result = result.view(1, result.size(0), result.size(1), result.size(2))
            result = scale(result).permute(0, 3, 1, 2)
            results = torch.cat((results, result))#.to(device)
        except:
            idxs = idxs[idxs!=i]
            continue

    return imgs_Y[idxs], results