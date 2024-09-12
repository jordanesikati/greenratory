import open3d as o3d
import numpy as np
import time
import torch
from pathlib import Path

class Trait:
    def __init__(self, file, step=0.01, save=False, gpu=False):
        self.file = file
        self.pcd = self.extract()
        self.radius = self.get_radius()[0]
        self.gpu = gpu
        self.step = step
        self.save = save
        self.device = torch.device("cuda") if self.gpu and torch.cuda.is_available() else torch.device("cpu")

    def extract(self):
        return o3d.io.read_point_cloud(str(self.file)) if self.file.lower().endswith('.ply')  else 'Incorrect File'

    def render(self):
        aabb = self.pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)

        obb = self.pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075)

        if self.save:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(self.pcd)
            # vis.add_geometry(aabb)
            vis.update_renderer()
            vis.poll_events()
            vis.capture_screen_image(Path(self.file).stem + str('trait.ply'))
            vis.destroy_window()

            ctr = vis.get_view_control()
            ctr.rotate(45.0, 450.0)  # adjust rotation as needed

            vis.update_renderer()

        else:
            o3d.visualization.draw_geometries([self.pcd, aabb, frame])
            #o3d.visualization.draw_geometries([self.pcd])

    def plot(self):
        if len(self.pcd.points) == 0:
            print('Not Enough Points')
        else:
            self.render()


    def get_radius(self):
        distances = np.linalg.norm(np.asarray(self.pcd.points) - self.pcd.get_center(), axis=1)
        return [np.min(distances), np.max(distances), np.mean(distances), np.std(distances)]

    def forward(self):

        started = time.time()

        # array to numpy and push to device
        P = np.asarray(self.pcd.points)
        P = torch.from_numpy(P).to(self.device)

        segments = []
        colors = []
        while P.shape[0] > 0:
            seed_point = P[0]  # step 2, select seed point from P
            Q = torch.unsqueeze(seed_point, 0)  # step 3, move seed point to Q
            Cn = torch.unsqueeze(seed_point, 0)  # step 4, create new cluster Cn
            color = torch.rand(3, device=self.device)  # step 5, assign random color to cluster
            i = 0
            while i < Q.shape[0]:
                pj = Q[i]  # step 6, select pj from Q
                j = 0
                while j < P.shape[0]:
                    pk = P[j]  # step 8, select pk from P
                    if torch.norm(pk - pj) < self.radius:  # step 9, check if pk is adjacent to pj
                        Q = torch.cat((Q, torch.unsqueeze(pk, 0)), dim=0)  # step 10, move pk from P to Q
                        Cn = torch.cat((Cn, torch.unsqueeze(pk, 0)), dim=0)  # add pk to Cn
                        P = torch.cat((P[:j], P[j + 1:]), dim=0)
                    else:
                        j += 1
                i += 1
            segments.append(Cn.cpu().numpy())  # add cluster Cn to segmented point cloud
            colors.extend([color.cpu().numpy()] * Cn.shape[0])  # assign color to points in cluster

        # rendering to open3d
        segments_o3d = o3d.geometry.PointCloud()
        segments_o3d.points = o3d.utility.Vector3dVector(np.concatenate(segments, axis=0))
        segments_o3d.colors = o3d.utility.Vector3dVector(np.array(colors))

        ended = time.time()

        print("Number of clusters:", len(segments_o3d))
        print("Processing time:", ended - started, "seconds")

        return segments_o3d


    def get_optimal_radius(self):

        # searching interval
        radii = np.arange(self.get_radius()[0], self.get_radius()[1] + self.step, self.step)
        clusters = []

        for r in radii:
            self.radius = r
            segment = self.forward()
            clusters.append(len(segment.points))

            print([radii[np.argmin(clusters)], min(clusters), max(clusters), np.mean(clusters), np.std(clusters)])

        return [radii[np.argmin(clusters)], min(clusters), max(clusters), np.mean(clusters), np.std(clusters)]





