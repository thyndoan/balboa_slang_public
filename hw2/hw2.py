import slangpy as spy
import device
import buffer
import window
import math
from . import hw2_scene
import time
import statistics
import numpy as np
import image
from PIL import Image
import os


PROJECTION = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, -1, 0]
], dtype=np.float32)


def screen_matrix(width, height, s, a):
    return np.array([
        [width/(2*s*a), 0, 0, width/2],
        [0, -height/(2*s), 0, height/2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)


def clip_projection_matrix(camera, a):
    return np.array([
        [1 / (camera.s * a), 0, 0, 0],
        [0, 1 / camera.s, 0, 0],
        [0, 0, -1, -camera.z_near],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def interpolate_transformation(transform_keyframes, t):
    # HW 2.4: implement this
    # interpolate the corresponding transformation parameters.
    #
    # Return a list of transformations in the same format
    # as the "transform" field in the scene JSON.
    keyframes = sorted(transform_keyframes, key=lambda k: k['time'])
    n = len(keyframes)

    if n == 0:
        return []
    if n == 1 or t <= keyframes[0]['time']:
        return keyframes[0]['transform']
    if t >= keyframes[-1]['time']:
        return keyframes[-1]['transform']

    def lerp(v0, v1, w):
        if isinstance(v0, dict):
            return {k: lerp(v0[k], v1[k], w) for k in v0}
        if isinstance(v0, (list, tuple)):
            return [lerp(x, y, w) for x, y in zip(v0, v1)]
        return (1 - w) * v0 + w * v1

    for k0, k1 in zip(keyframes[:-1], keyframes[1:]):
        t0, t1 = k0['time'], k1['time']
        if t0 <= t <= t1:
            w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            result = []
            for tr0, tr1 in zip(k0['transform'], k1['transform']):
                name = next(iter(tr0))
                result.append({name: lerp(tr0[name], tr1[name], w)})
            return result

    raise ValueError("Failed to find keyframe interval")


def cam_to_world_at(camera, t):
    if len(camera.transform_keyframes) > 0:
        return hw2_scene.compose_transformations(
            interpolate_transformation(camera.transform_keyframes, t))
    return camera.cam_to_world


def model_matrix_at(mesh, t):
    if len(mesh.transform_keyframes) > 0:
        return hw2_scene.compose_transformations(
            interpolate_transformation(mesh.transform_keyframes, t))
    return mesh.model_matrix


def time_from_args(args, scene):
    return (getattr(args, 'time', 0.0) or 0.0) % scene.duration


def run_app(args, slang_device, color_image, render_frame, t=0.0):
    if args.output is None:
        app = window.App(slang_device, color_image)

        def update(t, vars):
            vars['time'] = t
            return vars

        def render(command_encoder, vars, img):
            render_frame(command_encoder, vars['time'])

        app.run({'time': 0.0}, render, update)
    else:
        command_encoder = slang_device.create_command_encoder()
        render_frame(command_encoder, t)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        slang_device.wait_for_submit(submit_id)
        pixels = color_image.to_numpy()
        image.save_png(args.output, pixels)


def project_mesh(mesh, MVP):
    ones = np.ones((mesh.vertices.shape[0], 1), dtype=mesh.vertices.dtype)
    vertices = np.concatenate([mesh.vertices, ones], axis=1) @ MVP.T
    vertices[:, :3] /= vertices[:, 3:4]
    return (np.ascontiguousarray(vertices[:, :2]),
            np.ascontiguousarray(vertices[:, 2]))


def render_triangles(command_encoder, compute_shader, color_image,
                     depth_image, items, background, s, a):
    command_encoder.clear_texture_float(
        color_image,
        clear_value=spy.float4(background[0], background[1],
                               background[2], 1.0)
    )
    command_encoder.clear_texture_float(
        depth_image,
        clear_value=spy.float4(float("inf"), 0.0, 0.0, 0.0)
    )
    for mesh, screen_vertices, inv_depth in items:
        for v0, v1, v2 in mesh.faces:
            sp0 = screen_vertices[v0]
            sp1 = screen_vertices[v1]
            sp2 = screen_vertices[v2]

            xs = [sp0[0], sp1[0], sp2[0]]
            ys = [sp0[1], sp1[1], sp2[1]]

            aabb_min = (math.floor(min(xs)), math.floor(min(ys)))
            aabb_max = (math.ceil(max(xs)), math.ceil(max(ys)))

            if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
                continue

            compute_shader.kernel.dispatch(
                thread_count=[aabb_max[0] - aabb_min[0] + 1,
                              aabb_max[1] - aabb_min[1] + 1,
                              1],
                command_encoder=command_encoder,
                color_image=color_image,
                depth_image=depth_image,
                sp0=sp0,
                sp1=sp1,
                sp2=sp2,
                inv_depth0=inv_depth[v0],
                inv_depth1=inv_depth[v1],
                inv_depth2=inv_depth[v2],
                aabb_min=aabb_min,
                c0=mesh.vertex_colors[v0],
                c1=mesh.vertex_colors[v1],
                c2=mesh.vertex_colors[v2],
                s=s,
                a=a
            )
            # Synchronize to avoid race conditions on the depth buffer
            command_encoder.global_barrier()


def hw2_1(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(
        slang_device, 'hw2/hw2_1.slang')
    img = buffer.create_texture_2d(slang_device, args.width, args.height)

    a = img.width / img.height

    def to_screen(p):
        # HW2.1: implement this
        x, y, z = p

        # perspective projection (turning 3d into 2d)
        xp = -x / z
        yp = -y / z

        # access "s" using args.s
        # screen mapping
        sx = img.width * ((xp + args.s * a) / (2 * args.s * a))
        sy = img.height * ((args.s - yp) / (2 * args.s))
        return (sx, sy)

    sp0 = to_screen(args.p0)
    sp1 = to_screen(args.p1)
    sp2 = to_screen(args.p2)

    # HW2.1: implement this
    # store all the x of the 3 projected vertices in xs
    # and all the y in ys
    xs = [sp0[0], sp1[0], sp2[0]]
    ys = [sp0[1], sp1[1], sp2[1]]

    # aabb = smallest rectangle contains all 3 projected vertices
    aabb_min = (math.floor(min(xs)), math.floor(min(ys)))
    aabb_max = (math.ceil(max(xs)), math.ceil(max(ys)))

    vars = {'sp0': sp0,
            'sp1': sp1,
            'sp2': sp2,
            'aabb_min': aabb_min,
            'color': args.color}

    def render(command_encoder, vars, img):
        # Clear the image to background color
        command_encoder.clear_texture_float(
            img, clear_value=spy.float4(0.5, 0.5, 0.5, 1.0))

        if aabb_min[0] > aabb_max[0] or aabb_min[1] > aabb_max[1]:
            # the entire triangle is outside the screen
            return

        # Launch the kernel
        compute_shader.kernel.dispatch(
            thread_count=[aabb_max[0] - aabb_min[0] + 1,
                          aabb_max[1] - aabb_min[1] + 1,
                          1],
            command_encoder=command_encoder,
            image=img,
            **vars,
        )

    if args.output == None:
        app = window.App(slang_device, img)
        app.run(vars, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, vars, img)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, img.to_numpy())


def hw2_2(args):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(
        slang_device, 'hw2/hw2_2.slang')
    color_image = buffer.create_texture_2d(
        slang_device, args.width, args.height, num_channels=4)
    depth_image = buffer.create_texture_2d(
        slang_device, args.width, args.height, num_channels=1)

    a = args.width / args.height

    mesh = hw2_scene.parse_triangle_mesh(args.mesh)

    def to_screen(p):
        # copied from hw2_1
        x, y, z = p

        # perspective projection (turning 3d into 2d)
        xp = -x / z
        yp = -y / z

        # access "s" using args.s
        # screen mapping
        sx = args.width * ((xp + args.s * a) / (2 * args.s * a))
        sy = args.height * ((args.s - yp) / (2 * args.s))
        return (sx, sy)

    # splat the triangles onto the screen
    screen_vertices = [to_screen(mesh.vertices[vid, :])
                       for vid in range(mesh.vertices.shape[0])]
    # (recall that camera-space depth is -z, which we assume is positive)
    inv_depth = [1.0 / -mesh.vertices[vid][2]
                 for vid in range(mesh.vertices.shape[0])]

    def render(command_encoder, vars, img):
        render_triangles(command_encoder, compute_shader, color_image,
                         depth_image,
                         [(mesh, screen_vertices, inv_depth)],
                         (0.5, 0.5, 0.5), args.s, a)

    if args.output == None:
        app = window.App(slang_device, color_image)
        app.run({}, render)
    else:
        command_encoder = slang_device.create_command_encoder()
        render(command_encoder, {}, color_image)
        submit_id = slang_device.submit_command_buffer(
            command_encoder.finish()
        )
        # explicit synchronization
        slang_device.wait_for_submit(submit_id)
        image.save_png(args.output, color_image.to_numpy())


def render_scene_cpu_projection(args, animate):
    slang_device = spy.create_device()
    compute_shader = device.load_compute_shader(
        slang_device, 'hw2/hw2_2.slang')
    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    color_image = buffer.create_texture_2d(
        slang_device, width, height, num_channels=4)
    depth_image = buffer.create_texture_2d(
        slang_device, width, height, num_channels=1)

    a = width / height
    camera_to_screen = screen_matrix(width, height, camera.s, a)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        cam_to_world = cam_to_world_at(
            camera, t) if animate else camera.cam_to_world
        world_to_screen = (camera_to_screen @ PROJECTION @
                           np.linalg.inv(cam_to_world))

        items = []
        for mesh in scene.meshes:
            model_matrix = model_matrix_at(
                mesh, t) if animate else mesh.model_matrix
            screen_vertices, inv_depth = project_mesh(
                mesh, world_to_screen @ model_matrix)
            items.append((mesh, screen_vertices, inv_depth))

        render_triangles(command_encoder, compute_shader, color_image,
                         depth_image, items, scene.background,
                         camera.s, a)

    run_app(args, slang_device, color_image, render_frame,
            time_from_args(args, scene) if animate else 0.0)


def hw2_3(args):
    render_scene_cpu_projection(args, animate=False)


def hw2_4(args):
    render_scene_cpu_projection(args, animate=True)


def hw2_5(args):
    slang_device = spy.create_device()
    shaders = {name: device.load_compute_shader(
        slang_device, 'hw2/hw2_5.slang', name)
        for name in ['transform_vertices', 'clear_buffers',
                     'rasterize_depth', 'rasterize_id', 'shade']}

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Preprocessing: merge all meshes into one list
    # so that we can launch them on GPU
    vertices_list = []
    faces_list = []
    colors_list = []
    # need to retain the offset list for animation
    vertex_offsets = []
    vertex_offset = 0
    for mesh in scene.meshes:
        vertices_list.append(mesh.vertices)
        # the indexing needs to be updated to the new array
        faces_list.append(mesh.faces + vertex_offset)
        colors_list.append(mesh.vertex_colors)
        vertex_offsets.append(vertex_offset)
        vertex_offset += len(mesh.vertices)
    vertices_host = np.concatenate(vertices_list)
    faces_host = np.concatenate(faces_list).astype(np.uint32)
    vertex_colors_host = np.concatenate(colors_list)
    num_vertices = vertices_host.shape[0]
    num_faces = faces_host.shape[0]

    # Allocate/upload buffer
    vertices = buffer.structured_typed_buffer_from_numpy(
        slang_device, vertices_host, 'float3')
    projected_vertices = spy.Tensor.empty(
        slang_device, shape=(num_vertices,), dtype='float3')
    faces = buffer.structured_typed_buffer_from_numpy(
        slang_device, faces_host, 'uint3')
    vertex_colors = buffer.structured_typed_buffer_from_numpy(
        slang_device, vertex_colors_host, 'float3')
    depth_buffer = spy.Tensor.empty(
        slang_device, shape=(width * height,), dtype='uint')
    triangle_id_buffer = spy.Tensor.empty(
        slang_device, shape=(width * height,), dtype='uint')
    color_image = buffer.create_texture_2d(
        slang_device, width, height, num_channels=4)

    camera_to_screen = screen_matrix(width, height, camera.s, a)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        world_to_screen = (camera_to_screen @ PROJECTION @
                           np.linalg.inv(cam_to_world_at(camera, t)))

        command_encoder.clear_texture_float(
            color_image,
            clear_value=spy.float4(scene.background[0],
                                   scene.background[1],
                                   scene.background[2],
                                   1.0)
        )

        # Transform the vertices per mesh because each mesh
        # has a different model matrix
        for mesh, offset in zip(scene.meshes, vertex_offsets):
            MVP = (world_to_screen @ model_matrix_at(mesh, t)
                   ).astype(np.float32)
            shaders['transform_vertices'].kernel.dispatch(
                thread_count=[mesh.vertices.shape[0], 1, 1],
                command_encoder=command_encoder,
                num_vertices=mesh.vertices.shape[0],
                vertex_offset=offset,
                MVP=spy.float4x4(MVP),
                vertices=vertices,
                projected_vertices=projected_vertices.storage
            )

        shaders['clear_buffers'].kernel.dispatch(
            thread_count=[width * height, 1, 1],
            command_encoder=command_encoder,
            num_pixels=width * height,
            depth_buffer=depth_buffer.storage,
            triangle_id_buffer=triangle_id_buffer.storage)

        command_encoder.global_barrier()

        raster_args = dict(
            num_triangles=num_faces,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            projected_vertices=projected_vertices.storage,
            depth_buffer=depth_buffer.storage
        )

        shaders['rasterize_depth'].kernel.dispatch(
            thread_count=[num_faces, 1, 1],
            command_encoder=command_encoder,
            **raster_args
        )

        command_encoder.global_barrier()

        shaders['rasterize_id'].kernel.dispatch(
            thread_count=[num_faces, 1, 1],
            command_encoder=command_encoder,
            triangle_id_buffer=triangle_id_buffer.storage,
            **raster_args
        )

        command_encoder.global_barrier()

        shaders['shade'].kernel.dispatch(
            thread_count=[width, height, 1],
            command_encoder=command_encoder,
            width=width,
            height=height,
            s=camera.s,
            a=a,
            faces=faces,
            vertex_colors=vertex_colors,
            projected_vertices=projected_vertices.storage,
            triangle_id_buffer=triangle_id_buffer.storage,
            image=color_image)

        command_encoder.global_barrier()

    run_app(args, slang_device, color_image, render_frame,
            time_from_args(args, scene))


def create_Kd_texture(slang_device, material, scene_dir):
    if 'texture' in material:
        filename = os.path.join(scene_dir, material['texture'])
        rgba = np.asarray(
            Image.open(filename).convert('RGBA'),
            dtype=np.uint8)
    else:
        color = material['Kd']
        rgba = np.array(
            [[[*color, 1.0]]], dtype=np.float32)
        rgba = np.uint8(np.round(np.clip(rgba, 0, 1) * 255))

    height, width = rgba.shape[:2]

    return slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba8_unorm_srgb,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(rgba, dtype=np.uint8, copy=True, order='C').ravel()
    )


def upload_mesh(slang_device, mesh, width, height, attributes):
    gpu_mesh = {name: buffer.structured_typed_buffer_from_numpy(
        slang_device, data, dtype)
        for name, (data, dtype) in attributes.items()}

    indices_host = np.ascontiguousarray(
        mesh.faces.reshape(-1), dtype=np.uint32)

    indices = slang_device.create_buffer(
        size=indices_host.nbytes,
        usage=spy.BufferUsage.index_buffer,
        data=indices_host
    )

    gpu_mesh['indices'] = indices

    gpu_mesh['render_state'] = spy.RenderState({
        'viewports': [spy.Viewport.from_size(width, height)],
        'scissor_rects': [spy.ScissorRect.from_size(width, height)],
        'index_buffer': spy.BufferOffsetPair(indices),
        'index_format': spy.IndexFormat.uint32,
    })

    return gpu_mesh


def create_render_objects(slang_device, program, scene, width, height):
    color_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.rgba32_float,
        usage=(spy.TextureUsage.render_target |
               spy.TextureUsage.shader_resource |
               spy.TextureUsage.copy_source)
    )
    depth_image = slang_device.create_texture(
        width=width,
        height=height,
        format=spy.Format.d32_float,
        usage=spy.TextureUsage.depth_stencil
    )
    color_view = color_image.create_view()
    depth_view = depth_image.create_view()

    pipeline = slang_device.create_render_pipeline(
        program=program,
        input_layout=None,
        primitive_topology=spy.PrimitiveTopology.triangle_list,
        targets=[spy.ColorTargetDesc({"format": color_image.format})],
        depth_stencil=spy.DepthStencilDesc({
            "format": spy.Format.d32_float,
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_func": spy.ComparisonFunc.less,
        }),
        rasterizer=spy.RasterizerDesc({"cull_mode": spy.CullMode.none}),
    )

    # The color and depth textures are cleared explicitly before the render pass.
    # LoadOp.load preserves those cleared values when the pass begins.
    render_pass_desc = spy.RenderPassDesc({
        'color_attachments': [
            spy.RenderPassColorAttachment({
                'view': color_view,
                'load_op': spy.LoadOp.load,
                'store_op': spy.StoreOp.store,
            })
        ],
        'depth_stencil_attachment': spy.RenderPassDepthStencilAttachment({
            'view': depth_view,
            'depth_load_op': spy.LoadOp.load,
            'depth_store_op': spy.StoreOp.store,
        }),
    })

    return (color_image, depth_image, color_view, depth_view,
            pipeline, render_pass_desc)


def clear_targets(command_encoder, color_image, depth_image, background):
    command_encoder.clear_texture_float(
        color_image,
        clear_value=spy.float4(background[0], background[1],
                               background[2], 1.0)
    )
    command_encoder.clear_texture_depth_stencil(
        depth_image,
        clear_depth=True,
        depth_value=1.0,
        clear_stencil=False
    )


def draw_mesh(render_pass, pipeline, mesh, gpu_mesh, uniforms):
    root = render_pass.bind_pipeline(pipeline)
    cursor = spy.ShaderCursor(root)

    for name, value in uniforms.items():
        cursor.find_field(name).write(value)

    render_pass.set_render_state(gpu_mesh['render_state'])

    render_pass.draw_indexed(spy.DrawArguments({
        "vertex_count": 3 * len(mesh.faces),
        "instance_count": 1,
        "start_index_location": 0,
    }))


def hw2_6(args):
    slang_device = spy.create_device()
    program = slang_device.load_program(
        "hw2/hw2_6.slang",
        ["vertex_main", "fragment_main"]
    )

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    # Upload meshes to GPU.
    gpu_meshes = [
        upload_mesh(slang_device, mesh, width, height, {
            'vertices': (mesh.vertices, 'float3'),
            'vertex_colors': (mesh.vertex_colors, 'float3'),
        })
        for mesh in scene.meshes
    ]

    (color_image, depth_image, color_view, depth_view,
     pipeline, render_pass_desc) = create_render_objects(
        slang_device, program, scene, width, height)

    # Clip-space projection matrix provided by the assignment.
    clip_projection = clip_projection_matrix(camera, a)

    def render_frame(command_encoder, t):
        t = t % scene.duration
        view_matrix = np.linalg.inv(
            cam_to_world_at(camera, t)).astype(np.float32)

        clear_targets(command_encoder, color_image, depth_image,
                      scene.background)
        render_pass = command_encoder.begin_render_pass(render_pass_desc)

        for mesh, gpu_mesh in zip(scene.meshes, gpu_meshes):
            MVP = (clip_projection @ view_matrix @
                   model_matrix_at(mesh, t)).astype(np.float32)
            draw_mesh(render_pass, pipeline, mesh, gpu_mesh, {
                'MVP': spy.float4x4(MVP),
                'vertices': gpu_mesh['vertices'],
                'vertex_colors': gpu_mesh['vertex_colors'],
            })

        render_pass.end()

    run_app(args, slang_device, color_image, render_frame,
            time_from_args(args, scene))


def hw2_7(args):
    slang_device = spy.create_device()
    program = slang_device.load_program(
        "hw2/hw2_7.slang",
        ["vertex_main", "fragment_main"]
    )

    scene = hw2_scene.parse_scene(args.scene)
    camera = scene.camera
    width = camera.resolution[0]
    height = camera.resolution[1]
    a = width / height

    gpu_meshes = [
        upload_mesh(slang_device, mesh, width, height, {
            'vertices': (mesh.vertices, 'float3'),
            'vertex_normals': (mesh.vertex_normals, 'float3'),
            'vertex_uvs': (mesh.vertex_uvs, 'float2'),
        })
        for mesh in scene.meshes
    ]

    (color_image, depth_image, color_view, depth_view,
     pipeline, render_pass_desc) = create_render_objects(
        slang_device, program, scene, width, height)

    clip_projection = clip_projection_matrix(camera, a)

    Kd_textures = [
        create_Kd_texture(slang_device,
                          mesh.material,
                          os.path.dirname(args.scene))
        for mesh in scene.meshes
    ]
    Kd_sampler = slang_device.create_sampler()

    def render_frame(command_encoder, t):
        t = t % scene.duration
        cam_to_world = cam_to_world_at(camera, t)
        view_matrix = np.linalg.inv(cam_to_world).astype(np.float32)

        clear_targets(command_encoder, color_image, depth_image,
                      scene.background)
        render_pass = command_encoder.begin_render_pass(render_pass_desc)

        for mesh, gpu_mesh, Kd_texture in zip(scene.meshes, gpu_meshes, Kd_textures):
            model_matrix = model_matrix_at(mesh, t).astype(np.float32)
            normal_matrix = np.asarray(
                np.linalg.inv(model_matrix[:3, :3]).T, dtype=np.float32)
            MVP = (clip_projection @ view_matrix @
                   model_matrix).astype(np.float32)

            draw_mesh(render_pass, pipeline, mesh, gpu_mesh, {
                'MVP': spy.float4x4(MVP),
                'model_matrix': spy.float4x4(model_matrix),
                'normal_matrix': spy.float3x3(normal_matrix),
                'vertices': gpu_mesh['vertices'],
                'vertex_normals': gpu_mesh['vertex_normals'],
                'vertex_uvs': gpu_mesh['vertex_uvs'],
                'light_dir': spy.float3(scene.light_dir),
                'light_color': spy.float3(scene.light_color),
                'camera_position': spy.float3(cam_to_world[:3, 3]),
                'Kd_texture': Kd_texture,
                'Kd_sampler': Kd_sampler,
                'Ks': spy.float3(mesh.material['Ks']),
                'exponent': mesh.material['exponent'],
            })

        render_pass.end()

    run_app(args, slang_device, color_image, render_frame,
            time_from_args(args, scene))
