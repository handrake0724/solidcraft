use solidcore_record::Vec3;

use super::*;

/// The GPU, or `None` with a note when the machine has no adapter (CI runners without a GPU),
/// unless a GPU is required ([`test_env`]). A device that cannot build the pipeline fails the
/// test: that is a shader the device rejects (as D3D12's FXC once did), not a missing GPU.
fn gpu() -> Option<TessGpu> {
    match TessGpu::headless(test_env::policy()) {
        Ok(g) => {
            eprintln!("GPU: {}", g.adapter_name());
            Some(g)
        }
        Err(e @ GpuError::Pipeline(_)) => panic!("{e}"),
        Err(e) => {
            test_env::skip_unless_required(test_env::required(), &e);
            None
        }
    }
}

fn ok<T, E: std::fmt::Debug>(r: std::result::Result<T, E>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{e:?}"),
    }
}

#[test]
fn the_hlsl_for_fxc_has_no_arrays_fxc_must_unroll_loops_for() {
    use naga::back::hlsl;
    let module = ok(naga::front::wgsl::parse_str(SHADER));
    let info = ok(naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default()).validate(&module));
    let options = hlsl::Options {
        shader_model: hlsl::ShaderModel::V5_1,
        force_loop_bounding: true,
        restrict_indexing: true,
        fake_missing_bindings: true,
        ..Default::default()
    };
    let pipeline = hlsl::PipelineOptions::default();
    let mut out = String::new();
    ok(hlsl::Writer::new(&mut out, &options, &pipeline).write(&module, &info, None));
    assert!(out.contains("void main("), "{out}");
    // Struct members that are arrays: `float n0_[8];` inside a `struct ... { };`.
    let mut in_struct = false;
    for line in out.lines() {
        let l = line.trim();
        if l.starts_with("struct ") && !l.ends_with(';') {
            in_struct = true;
        } else if l.starts_with("};") {
            in_struct = false;
        } else if in_struct {
            assert!(!l.contains('['), "an array inside a struct: {l}");
        }
        assert!(!l.contains("]["), "a two-dimensional array: {l}");
    }
}

fn check_empty_tessellation(gpu: &TessGpu, plan: &TessPlan) {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let evaluated = ok(gpu.evaluate(plan));
    assert!(evaluated.positions.is_empty());
    assert!(evaluated.normals.is_empty());
    assert!(evaluated.triangles.is_empty());

    let mesh = ok(gpu.prepare(plan));
    assert_eq!(mesh.vertex_count, 0);
    assert_eq!(mesh.index_count, 0);
    assert!(mesh._inputs[1].size() >= std::mem::size_of::<[u32; SURFACE_WORDS]>() as u64);
    assert!(mesh._inputs[2].size() >= std::mem::size_of::<f32>() as u64);
    assert!(mesh.parameters.size() >= std::mem::size_of::<[u32; VERTEX_WORDS]>() as u64);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    gpu.record(&mut encoder, &mesh);
    gpu.queue.submit([encoder.finish()]);
    ok(gpu.device.poll(wgpu::PollType::wait_indefinitely()));
    let error = pollster::block_on(scope.pop());
    assert!(error.is_none(), "{error:?}");
}

#[test]
fn tessellation_without_vertices_does_not_dispatch() {
    let Some(gpu) = gpu() else { return };
    let plan = TessPlan {
        surfaces: vec![EvalSurface::Plane {
            origin: solidcore_tess::vec3(Vec3::ZERO),
            du: solidcore_tess::vec3(Vec3::X),
            dv: solidcore_tess::vec3(Vec3::Y),
        }],
        faces: vec![solidcore_tess::FacePlan { surface: Some(0), ..Default::default() }],
        ..Default::default()
    };
    check_empty_tessellation(&gpu, &plan);
}

#[test]
fn tessellation_without_surfaces_does_not_dispatch() {
    let Some(gpu) = gpu() else { return };
    check_empty_tessellation(&gpu, &TessPlan::default());
}

/// The curvature map's per-pixel evaluation (`view/exact.wgsl`) gives the CPU's exact curvature
/// (`EvalSurface::curvature`) on a rational, doubly curved patch far from the origin (a ship's
/// stern, 200 m along): the control points taken relative to their window keep f32's digits.
#[test]
fn the_curvature_map_evaluates_the_surface_as_the_cpu_does() {
    let Some(gpu) = gpu() else { return };
    let (device, queue) = (&gpu.device, &gpu.queue);
    if device.limits().max_storage_buffers_per_shader_stage < 4 {
        eprintln!("skipping: no storage buffers");
        return;
    }
    // Degree 3 × 2, 5 × 4 control points, weights off 1, a dome with a crease-free ridge.
    let (nu, nv) = (5usize, 4usize);
    let origin = [200_000.0, 50_000.0, 10_000.0];
    let mut ctrl = Vec::new();
    for j in 0..nv {
        for i in 0..nu {
            let (x, y) = (i as f64 * 3000.0, j as f64 * 2500.0);
            let z = 900.0 * ((i as f64 - 2.0).powi(2) * 0.4 + (j as f64 - 1.5).powi(2) * 0.7) + 250.0 * (i * j) as f64;
            let w = 1.0 + 0.15 * ((i + 2 * j) % 3) as f64;
            ctrl.push([(x + origin[0]) * w, (y + origin[1]) * w, (z + origin[2]) * w, w]);
        }
    }
    let knots_u = vec![0.0, 0.0, 0.0, 0.0, 4000.0, 9000.0, 9000.0, 9000.0, 9000.0];
    let knots_v = vec![0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0];
    let s = EvalSurface::nurbs([3, 2], [knots_u, knots_v], [nu, nv], ctrl);
    let plan = TessPlan { surfaces: vec![s.clone()], ..Default::default() };
    let packed = ok(pack(&[&plan]));
    let mut q: Vec<[f32; 2]> = Vec::new();
    for a in 0..9 {
        for b in 0..9 {
            q.push([9000.0 * (a as f32 + 0.5) / 9.0, (b as f32 + 0.5) / 9.0]);
        }
    }
    let source = format!(
        "{}\n{}\n{}\n{}",
        crate::view::SHADER,
        crate::pbr::SOURCE,
        crate::view::EXACT_SHADER,
        "@group(3) @binding(2) var<storage, read> ex_q: array<vec2<f32>>;
@group(3) @binding(3) var<storage, read_write> ex_k: array<f32>;
@compute @workgroup_size(1)
fn ex_test(@builtin(global_invocation_id) g: vec3<u32>) { ex_k[g.x] = ex_curvature(ex_q[g.x], 0u); }"
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(source.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("ex_test"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bytes = |v: Vec<u32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let storage = |contents: Vec<u8>, usage| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: &contents, usage });
    let surfaces = storage(bytes(packed.surfaces.clone()), wgpu::BufferUsages::STORAGE);
    let data = storage(bytes(packed.data.iter().map(|x| x.to_bits()).collect()), wgpu::BufferUsages::STORAGE);
    let params = storage(bytes(q.iter().flat_map(|p| [p[0].to_bits(), p[1].to_bits()]).collect()), wgpu::BufferUsages::STORAGE);
    let size = (q.len() * 4) as u64;
    let out = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let read = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(3),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: surfaces.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: data.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
        ],
    });
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(3, &bind, &[]);
        pass.dispatch_workgroups(q.len() as u32, 1, 1);
    }
    enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
    queue.submit([enc.finish()]);
    read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    ok(device.poll(wgpu::PollType::wait_indefinitely()));
    let got: Vec<f32> = ok(read.slice(..).get_mapped_range()).as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let mut worst = 0.0f64;
    for (p, k) in q.iter().zip(&got) {
        let want = s.curvature(f64::from(p[0]), f64::from(p[1])).unwrap_or(-1.0);
        assert!(want > 0.0, "a doubly curved patch: {want} at {p:?}");
        let err = (f64::from(*k) - want).abs() / want;
        worst = worst.max(err);
        assert!(err < 1e-3, "GPU {k} against CPU {want} at {p:?}");
    }
    eprintln!("worst relative difference {worst:.2e}");
}
