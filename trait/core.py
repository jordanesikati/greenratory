import open3d as o3d
import numpy as np
import time
import torch
import plyfile
from scipy.spatial import Delaunay, ConvexHull
from pathlib import Path

class Processing:
    def __init__(self, file, texture=None, crop=False, points=1000):
        self.file = file
        self.texture = texture
        self.crop = crop
        self.points = points

    def to_ply(self):
        return o3d.io.read_triangle_mesh(self.file).sample_points_uniformly(number_of_points=self.points) if self.file.lower().endswith('.obj') else 'Incorrect File'

    def save_ply(self):
        o3d.io.write_point_cloud(Path(self.file).stem + str('_saved.ply'), self.to_ply())

    def render_obj(self):
        mesh = o3d.io.read_triangle_mesh(self.file)
        # mesh.material = o3d.io.read_mtl(self.texture)
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        if self.crop:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)
            vis.update_renderer()
            vis.poll_events()
            vis.capture_screen_image(Path(self.file).stem + str('_obj.png'))
            vis.destroy_window()

            ctr = vis.get_view_control()
            ctr.rotate(45.0, 450.0)
            vis.update_renderer()

        else:
            # vis.get_render_option().load_from_json("viewpoint.json")
            vis.run()
            vis.destroy_window()

    def plot_obj(self):
        if self.file.lower().endswith('.obj'): self.render_obj()


class Volumetric:
    def __init__(self, file):
        self.file = file
        self.pcd = self.extract()

    def extract(self):
        return np.column_stack((plyfile.PlyData().read(self.file)['vertex']['x'],
                                plyfile.PlyData().read(self.file)['vertex']['y'],
                                plyfile.PlyData().read(self.file)['vertex']['z'])
                               ) if self.file.lower().endswith('.ply') else 'Incorrect File'

    def get_volume(self):
        tv = []
        for i in Delaunay(self.pcd).simplices: # tetrahedrons indices
            tvi = abs(np.linalg.det(np.vstack((self.pcd[i].T, np.ones(4)))))  # tetrahedron volume
            tv.append(tvi)

        return np.sum(tv) * 1000000 / 6.0 # cm³

    def get_area(self):
       return ConvexHull(self.pcd[:, :2]).area * 10000 # xy projections


class Trait:
    def __init__(self, file, step=0.01, default=False, crop=False, gpu=False):
        self.file = file
        self.pcd = self.extract()
        self.radius = self.get_radius()[0]
        self.gpu = gpu
        self.step = step
        self.crop = crop
        self.default = default
        self.device = torch.device("cuda") if self.gpu and torch.cuda.is_available() else torch.device("cpu")

    def extract(self):
        return o3d.io.read_point_cloud(str(self.file)) if self.file.lower().endswith('.ply')  else 'Incorrect File'

    def render_ply(self):
        aabb = self.pcd.get_axis_aligned_bounding_box()
        aabb.color = (1, 0, 0)

        obb = self.pcd.get_oriented_bounding_box()
        obb.color = (0, 1, 0)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.075)

        if self.crop:
            self.pcd = self.forward()
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(self.pcd)
            # vis.add_geometry(aabb)
            vis.update_renderer()
            vis.poll_events()
            vis.capture_screen_image(Path(self.file).stem + str('_trait.png'))
            vis.destroy_window()

            ctr = vis.get_view_control()
            ctr.rotate(45.0, 450.0)  # adjust rotation as needed

            vis.update_renderer()

        elif not self.default:
            self.pcd = self.forward()
            o3d.visualization.draw_geometries([self.pcd, aabb, frame])
            # o3d.visualization.draw_geometries([self.pcd])

        else:
            o3d.visualization.draw_geometries([self.pcd, aabb, frame])
            #o3d.visualization.draw_geometries([self.pcd])

    def plot_ply(self):
        if len(self.pcd.points) == 0:
            print('Not Enough Points')
        else:
            self.render_ply()

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

        print("Number of clusters:", len(segments))
        print("Processing time:", ended - started, "seconds")

        return segments_o3d

    def min(self):
        # lowest point
        return [np.min(np.asarray(self.pcd.points)[:, 2]), np.argmin(np.asarray(self.pcd.points)[:, 2])]

    def max(self):
        # highest point
        return [np.max(np.asarray(self.pcd.points)[:, 2]), np.argmax(np.asarray(self.pcd.points)[:, 2])]

    def distance(self):
        # from lowest to highest point
        return np.linalg.norm(np.asarray(self.pcd.points)[self.max()[1]] -
                              np.asarray(self.pcd.points)[self.min()[1]]) * 100

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

